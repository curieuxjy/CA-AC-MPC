#!/usr/bin/env python3
"""Analytic Policy Gradient (APG) trainer for the actor-MPC on Brax (Phase 7).

The DiffMimic idea, adapted to our actor-MPC + locomotion-first scope:

  * Roll the policy out in the *differentiable* Brax sim for a short window H.
  * Backprop the windowed return through the sim AND through the differentiable
    MPC into the actor's cost network -- a true analytic policy gradient (no
    PPO, no score-function estimator).
  * DReplay: every ``dreplay_interval`` windows, reset the sim to the
    demonstration/initial state; otherwise just detach the carried state. This
    truncates the gradient horizon, the key trick that keeps long-horizon
    differentiable rollouts from exploding / sticking in bad minima.

For locomotion we differentiate Brax's native reward (forward progress - ctrl
cost) -- the natural analog of DiffMimic's imitation loss. A ``reference_fn``
hook is left open for future mocap imitation (state-matching loss).

EXPERIMENTAL: depends on the JAX<->PyTorch gradient bridge (envs/jax_torch_bridge.py).
If the boundary proves unstable, a JAX-native actor-MPC is the cleaner path
(docs/brax_port_plan.md, Phase 7).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
for p in (
    ROOT,
    ROOT / "acmpc_public-master" / "mpc.pytorch",
    ROOT / "acmpc_public-master" / "training_modules",
):
    sys.path.insert(0, str(p))

from utils.train_config import apply_overrides, load_yaml  # noqa: E402


def _set_env_vars(cfg: dict, device: str) -> None:
    os.environ["ACMPC_T"] = str(int(cfg.get("mpc_horizon", 3)))
    os.environ["ACMPC_MPC_MAX_ITER"] = str(int(cfg.get("mpc_max_iter", 5)))
    os.environ["ACMPC_MPC_BACKEND"] = "diffmpc"
    os.environ["ACMPC_MPC_GRAD_MODE"] = "diff"
    if device != "auto":
        os.environ["ACMPC_MPC_DEVICE"] = device
    for cfg_key, env_key in (
        ("mpc_dt", "ACMPC_BRAX_DT"),
        ("mpc_ctrl_scale", "ACMPC_BRAX_CTRL_SCALE"),
        ("mpc_range_q", "ACMPC_BRAX_RANGE_Q"),
        ("mpc_range_p", "ACMPC_BRAX_RANGE_P"),
    ):
        if cfg.get(cfg_key) is not None:
            os.environ[env_key] = str(cfg.get(cfg_key))


def train_apg(cfg: dict) -> None:
    device = str(cfg.get("device", "auto"))
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    _set_env_vars(cfg, device)

    from envs.diff_brax_env import DifferentiableBraxEnv
    from mlp_mpc_policy_brax import CustomNetworkBrax

    seed = int(cfg.get("seed", 0))
    th.manual_seed(seed)

    env_cfg = cfg.get("env", {}) or {}
    bk = dict(env_cfg.get("kwargs", {}) or {})
    apg = cfg.get("apg", {}) or {}
    H = int(apg.get("window", 16))
    gamma = float(apg.get("gamma", 0.99))
    lr = float(apg.get("learning_rate", 3e-4))
    max_grad_norm = float(apg.get("max_grad_norm", 1.0))
    dreplay_interval = int(apg.get("dreplay_interval", 4))
    total_timesteps = int(apg.get("total_timesteps", 2_000_000))
    log_every = int(apg.get("log_every", 20))

    n_envs = int((cfg.get("vec_env", {}) or {}).get("n_envs", 64))

    env = DifferentiableBraxEnv(
        robot=str(bk.get("robot", "halfcheetah")),
        n_envs=n_envs,
        episode_length=int(bk.get("episode_length", 1000)),
        action_repeat=int(bk.get("action_repeat", 1)),
        seed=seed,
        backend=bk.get("brax_backend", None),
        device=device,
    )

    actor = CustomNetworkBrax(env.task_obs_dim, n_ctrl=env.n_act).to(device)
    opt = th.optim.Adam(actor.parameters(), lr=lr)

    env.reset()
    steps_per_window = H * n_envs
    n_windows = max(1, total_timesteps // steps_per_window)
    print(
        f"[APG] device={device} robot={bk.get('robot','halfcheetah')} n_envs={n_envs} "
        f"H={H} windows={n_windows} dreplay_interval={dreplay_interval}"
    )

    ema_ret = None
    for w in range(n_windows):
        opt.zero_grad(set_to_none=True)
        obs_full, mpc_state = env.current_obs()

        window_return = th.zeros((), device=device)
        disc = 1.0
        for _h in range(H):
            task = obs_full[:, env.mpc_state_dim:]
            action = actor.forward_actor(task, mpc_state)
            obs_full, reward, _done, mpc_state = env.step(action)
            window_return = window_return + disc * reward.mean()
            disc *= gamma

        loss = -window_return  # maximize return (locomotion); swap for reference_fn later
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        opt.step()

        # DReplay: full reset to demonstration/initial state, else truncate window.
        if (w + 1) % dreplay_interval == 0:
            env.reset_window()
        else:
            env.detach_state()

        ret_val = float(window_return.detach().cpu())
        ema_ret = ret_val if ema_ret is None else 0.98 * ema_ret + 0.02 * ret_val
        if (w % log_every) == 0:
            print(
                f"[APG] window {w}/{n_windows}  return={ret_val:.3f}  ema={ema_ret:.3f}  "
                f"grad_norm={float(grad_norm):.3f}"
            )

    out_dir = Path(str((cfg.get("logging", {}) or {}).get("log_dir", "runs/apg_brax")))
    out_dir.mkdir(parents=True, exist_ok=True)
    th.save(actor.state_dict(), out_dir / "actor.pt")
    print(f"[APG] done. actor saved to {out_dir / 'actor.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train actor-MPC with APG on a Brax robot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)
    train_apg(cfg)


if __name__ == "__main__":
    main()
