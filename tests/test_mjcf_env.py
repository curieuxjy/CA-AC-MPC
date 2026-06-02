#!/usr/bin/env python3
"""Smoke test for the generic MJCF/MJX locomotion env (complex-robot port).

Loads Brax's bundled humanoid via the generic MJCF env, builds a batched MJX
env, and checks reset/step shapes + the actuated-joint state layout the
template MPC expects. Needs jax + brax + mujoco (mjx). Run:

    python tests/test_mjcf_env.py
    # custom robot:
    MJCF=/path/to/robot.xml python tests/test_mjcf_env.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_mjcf_locomotion_env():
    import jax
    import envs.mjcf_env  # noqa: F401  OUR package: registers 'mjcf_locomotion'
    import brax.envs as brax_envs

    mjcf_path = os.environ.get("MJCF", "brax_humanoid")
    n = 2
    env = brax_envs.create(
        "mjcf_locomotion",
        episode_length=10,
        auto_reset=False,
        batch_size=n,
        backend="mjx",
        mjcf_path=mjcf_path,
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(0))
    obs_dim = int(state.obs.shape[-1])
    n_act = int(env.action_size)
    assert state.obs.shape == (n, obs_dim)
    print(f"[ok] reset: obs_dim={obs_dim} n_act={n_act}")

    action = np.zeros((n, n_act), dtype=np.float32)
    state = step(state, jax.numpy.asarray(action))
    assert state.reward.shape == (n,)
    assert state.done.shape == (n,)
    assert np.isfinite(np.asarray(state.obs)).all()
    print(f"[ok] step: reward{tuple(state.reward.shape)} done{tuple(state.done.shape)} finite obs")

    # Actuated-joint MPC state layout (last n_act of q/qd) used by the adapter.
    q = np.asarray(state.pipeline_state.q if hasattr(state.pipeline_state, "q")
                   else state.pipeline_state.qpos)
    qd = np.asarray(state.pipeline_state.qd if hasattr(state.pipeline_state, "qd")
                    else state.pipeline_state.qvel)
    mpc_state = np.concatenate([q[..., -n_act:], qd[..., -n_act:]], axis=-1)
    assert mpc_state.shape == (n, 2 * n_act)
    print(f"[ok] actuated MPC-state dim = {2 * n_act} (= 2*n_act)")


if __name__ == "__main__":
    test_mjcf_locomotion_env()
    print("\nMJCF/MJX env smoke test passed.")
