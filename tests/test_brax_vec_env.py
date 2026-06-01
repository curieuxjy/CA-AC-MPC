#!/usr/bin/env python3
"""Phase-1 smoke test for the Brax -> SB3 VecEnv adapter.

Runs on CPU; no CUDA required. Requires jax, brax, gym, and the vendored SB3
fork on the path. NOT run automatically (the local machine lacks these deps);
execute manually once a compatible environment exists:

    python tests/test_brax_vec_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "stable-baselines3-acmpc-acmpc"))

from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from envs.brax_vec_env import BraxVecEnvAdapter  # noqa: E402
from utils.train_support import ResetWithRawStateWrapper  # noqa: E402

N = 4
EP_LEN = 16


def test_adapter_basic():
    env = BraxVecEnvAdapter(robot="halfcheetah", n_envs=N, episode_length=EP_LEN, seed=0)
    obs_dim = env.observation_space.shape[0]
    assert env.num_envs == N
    assert env.mpc_state_dim > 0
    assert obs_dim > env.mpc_state_dim  # there is a non-empty task-obs tail

    obs = env.reset()
    assert isinstance(obs, np.ndarray) and obs.shape == (N, obs_dim) and obs.dtype == np.float32

    a = np.asarray(env.action_space.sample())[None].repeat(N, axis=0)
    obs2, rew, done, infos = env.step(a)
    assert obs2.shape == (N, obs_dim)
    assert rew.shape == (N,) and done.shape == (N,)
    assert len(infos) == N
    print(f"[ok] basic step: obs_dim={obs_dim} mpc_state_dim={env.mpc_state_dim}")

    # Run past the episode length to trigger done + episode/terminal info.
    saw_episode_info = False
    for _ in range(EP_LEN + 4):
        obs2, rew, done, infos = env.step(a)
        for i in range(N):
            if done[i]:
                assert "episode" in infos[i] and "r" in infos[i]["episode"]
                assert "terminal_observation" in infos[i]
                saw_episode_info = True
    assert saw_episode_info, "episode never terminated within episode_length"
    print("[ok] auto-reset, episode info, terminal_observation")
    env.close()


def test_wrapper_chain():
    """The repo chain: adapter -> VecNormalize -> ResetWithRawStateWrapper."""
    venv = BraxVecEnvAdapter(robot="halfcheetah", n_envs=N, episode_length=EP_LEN, seed=1)
    state_dim = venv.mpc_state_dim
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv = ResetWithRawStateWrapper(venv, state_dim=state_dim)

    obs, state = venv.reset()
    assert state.shape == (N, state_dim)
    # The raw (un-normalized) prefix must match the MPC state slice.
    raw = venv.get_original_obs()
    assert np.allclose(raw[:, :state_dim], state, atol=1e-5)
    print("[ok] VecNormalize + ResetWithRawStateWrapper contract")
    venv.close()


if __name__ == "__main__":
    test_adapter_basic()
    test_wrapper_chain()
    print("\nAll Phase-1 smoke tests passed.")
