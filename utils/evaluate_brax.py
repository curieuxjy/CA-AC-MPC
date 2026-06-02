#!/usr/bin/env python3
"""Phase-6 evaluation for Brax-backed policies (mlp_only / acmpc_brax).

Kept separate from the drone evaluator (``evaluate_acmpc2.py``), which is tied
to gates/tracks/trajectory plots. This reuses the backend-agnostic
``evaluate_policy_vec`` (return metric) and adds Brax env construction, a
step-time measurement (so acmpc_brax vs mlp_only MPC overhead is visible), and
an optional Brax HTML render.

Usage:
    python utils/evaluate_brax.py \
        --model-path runs/acmpc_brax_halfcheetah/halfcheetah/model.zip \
        --robot halfcheetah --policy-type acmpc_brax --mpc-horizon 3 \
        --episodes 30 [--render out.html]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "mpc.pytorch"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "diff_mpc_drones"))
sys.path.insert(0, str(ROOT / "differentialMPCPerformance"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "training_modules"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "stable-baselines3-acmpc-acmpc"))

from stable_baselines3 import PPO  # noqa: E402

from utils.train_support import (  # noqa: E402
    _load_checkpoint_metadata,
    _make_vec_env_from_cfg,
    evaluate_policy_vec,
    resolve_policy_class,
)


def _set_mpc_env_vars(*, mpc_horizon: int, mpc_max_iter: int, brax_overrides: Dict[str, str]) -> None:
    """The acmpc_brax policy reads these at construction time."""
    os.environ["ACMPC_T"] = str(int(mpc_horizon))
    os.environ["ACMPC_MPC_MAX_ITER"] = str(int(mpc_max_iter))
    os.environ["ACMPC_MPC_BACKEND"] = "diffmpc"
    for k, v in brax_overrides.items():
        if v is not None:
            os.environ[k] = str(v)


def _measure_step_time(model: PPO, env, state_dim: int, steps: int = 200) -> Dict[str, float]:
    obs, _state = env.reset()
    n = env.num_envs

    def _drone_state(o):
        raw = env.get_original_obs() if hasattr(env, "get_original_obs") else o
        return raw[:, :state_dim] if isinstance(raw, np.ndarray) else raw

    # Warmup (JIT / CUDA graph compile).
    for _ in range(5):
        a, _ = model.policy.predict(obs, drone_state=_drone_state(obs), deterministic=True)
        obs, *_ = env.step(a)

    t0 = time.perf_counter()
    for _ in range(steps):
        a, _ = model.policy.predict(obs, drone_state=_drone_state(obs), deterministic=True)
        obs, *_ = env.step(a)
    dt = time.perf_counter() - t0
    return {
        "ms_per_vec_step": 1000.0 * dt / steps,
        "env_fps": n * steps / dt,
    }


def evaluate_brax_model(
    model_path: str,
    *,
    robot: str = "halfcheetah",
    policy_type: str = "acmpc_brax",
    mpc_horizon: int = 3,
    mpc_max_iter: int = 5,
    episode_length: int = 1000,
    n_eval_envs: int = 16,
    episodes: int = 30,
    deterministic: bool = True,
    device: str = "auto",
    brax_backend: str = "generalized",
    mpc_state_mode: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    vecnorm_path: Optional[str] = None,
    seed: int = 0,
    render: Optional[str] = None,
    brax_dt: Optional[float] = None,
    brax_ctrl_scale: Optional[float] = None,
) -> Dict[str, float]:
    if mpc_state_mode is None:
        mpc_state_mode = "actuated" if policy_type == "acmpc_brax" else "full"

    _set_mpc_env_vars(
        mpc_horizon=mpc_horizon,
        mpc_max_iter=mpc_max_iter,
        brax_overrides={
            "ACMPC_BRAX_DT": brax_dt,
            "ACMPC_BRAX_CTRL_SCALE": brax_ctrl_scale,
        },
    )

    # Auto-resolve VecNormalize stats saved alongside the checkpoint.
    if vecnorm_path is None:
        cand = Path(str(model_path) + ".vecnorm.pkl")
        if cand.exists():
            vecnorm_path = str(cand)

    env_kwargs = {
        "robot": robot,
        "episode_length": int(episode_length),
        "action_repeat": 1,
        "brax_backend": brax_backend,
        "mpc_state_mode": mpc_state_mode,
    }
    if mjcf_path is not None:
        env_kwargs["mjcf_path"] = mjcf_path
        if brax_backend == "generalized":
            env_kwargs["brax_backend"] = "mjx"
    env = _make_vec_env_from_cfg(
        track=robot,
        env_kwargs=env_kwargs,
        seed=seed,
        n_envs=n_eval_envs,
        vec_type="dummy",
        normalize_obs=(vecnorm_path is not None),
        clip_obs=10.0,
        log_dir=".",
        state_dim=2,  # overridden internally by the adapter's mpc_state_dim
        vecnorm_stats_path=vecnorm_path,
        env_backend="brax",
    )
    state_dim = int(getattr(env, "mpc_state_dim", 2))

    policy_class = resolve_policy_class(policy_type, mpc_backend="diffmpc")
    model = PPO.load(
        str(model_path),
        env=env,
        device=device,
        custom_objects={"policy_class": policy_class},
    )

    mean_r, std_r = evaluate_policy_vec(
        model, env, n_episodes=episodes, state_dim=state_dim, deterministic=deterministic
    )
    timing = _measure_step_time(model, env, state_dim=state_dim)

    print("\n=== Brax evaluation ===")
    print(f"  robot={robot} policy={policy_type} horizon={mpc_horizon} envs={n_eval_envs}")
    print(f"  return: mean={mean_r:.2f}  std={std_r:.2f}  (n={episodes})")
    print(f"  step: {timing['ms_per_vec_step']:.2f} ms/vec-step  ({timing['env_fps']:.0f} env-fps)")

    if render:
        try:
            _render_html(robot, episode_length, brax_backend, model, state_dim, render, seed)
            print(f"  rendered: {render}")
        except Exception as exc:  # rendering is best-effort
            print(f"  [render] skipped: {exc}")

    env.close()
    return {"return_mean": mean_r, "return_std": std_r, **timing}


def _render_html(robot, episode_length, brax_backend, model, state_dim, out_path, seed):
    """Best-effort single-episode HTML render via Brax's html exporter."""
    import jax
    from brax import envs
    from brax.io import html

    base = envs.create(robot, episode_length=episode_length, auto_reset=False, backend=brax_backend)
    reset = jax.jit(base.reset)
    step = jax.jit(base.step)
    state = reset(jax.random.PRNGKey(seed))
    rollout = [state.pipeline_state]
    # Single-env greedy rollout using the trained policy (numpy bridge).
    n_act = int(base.action_size)
    for _ in range(episode_length):
        obs = np.asarray(state.obs, dtype=np.float32)[None]
        # Build the [mpc_state | task_obs] layout the policy expects.
        q = np.asarray(state.pipeline_state.q if hasattr(state.pipeline_state, "q")
                       else state.pipeline_state.qpos)
        qd = np.asarray(state.pipeline_state.qd if hasattr(state.pipeline_state, "qd")
                        else state.pipeline_state.qvel)
        mpc_state = np.concatenate([q[-n_act:], qd[-n_act:]])[None].astype(np.float32)
        full_obs = np.concatenate([mpc_state, obs], axis=1)
        a, _ = model.policy.predict(full_obs, drone_state=mpc_state, deterministic=True)
        state = step(state, jax.numpy.asarray(a[0]))
        rollout.append(state.pipeline_state)
        if bool(state.done):
            break
    with open(out_path, "w") as f:
        f.write(html.render(base.sys, rollout))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Brax-backed CA-AC-MPC policy.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--robot", default="halfcheetah")
    p.add_argument("--policy-type", default="acmpc_brax", choices=["acmpc_brax", "mlp_only"])
    p.add_argument("--mpc-horizon", type=int, default=None, help="defaults to checkpoint metadata or 3")
    p.add_argument("--mpc-max-iter", type=int, default=5)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--n-eval-envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--brax-backend", default="generalized")
    p.add_argument("--mjcf-path", default=None, help="custom MJCF (complex robots); e.g. brax_humanoid")
    p.add_argument("--vecnorm-path", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", default=None, help="optional output .html path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    mpc_horizon = args.mpc_horizon
    if mpc_horizon is None:
        meta = _load_checkpoint_metadata(args.model_path)
        sig = meta.get("signature") if isinstance(meta, dict) else None
        mpc_horizon = int(sig.get("mpc_horizon")) if isinstance(sig, dict) and sig.get("mpc_horizon") else 3
    evaluate_brax_model(
        args.model_path,
        robot=args.robot,
        policy_type=args.policy_type,
        mpc_horizon=mpc_horizon,
        mpc_max_iter=args.mpc_max_iter,
        episode_length=args.episode_length,
        n_eval_envs=args.n_eval_envs,
        episodes=args.episodes,
        device=args.device,
        brax_backend=args.brax_backend,
        mjcf_path=args.mjcf_path,
        vecnorm_path=args.vecnorm_path,
        seed=args.seed,
        render=args.render,
    )


if __name__ == "__main__":
    main()
