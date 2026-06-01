"""Differentiable Brax env for Analytic Policy Gradient (Phase 7 / APG).

Unlike ``BraxVecEnvAdapter`` (Phase 1, a non-differentiable SB3 VecEnv for PPO),
this exposes a *differentiable* step: gradients flow from the returned
observation / reward back through the Brax physics into the action (and thus,
when the action comes from the actor-MPC, into the policy parameters).

Mechanism: the full Brax ``State`` pytree is flattened to a single vector with
``ravel_pytree`` and carried across steps as a torch tensor (with grad). Each
step calls ``jax.vjp`` through ``env.step`` via ``jax2torch``. Carrying the
flat state preserves *multi-step* gradients (action_t affecting loss at t+k),
which is what APG needs; DReplay (in the trainer) periodically detaches/​resets
it to keep the gradient window short and stable.

EXPERIMENTAL -- see jax_torch_bridge.py caveats. Observation layout matches the
Phase-1 adapter: ``[mpc_state (actuated [q,qd]) | task_obs]``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from .brax_vec_env import ROBOT_SPECS
from .jax_torch_bridge import jax2torch


class DifferentiableBraxEnv:
    def __init__(
        self,
        robot: str = "halfcheetah",
        *,
        n_envs: int = 64,
        episode_length: int = 1000,
        action_repeat: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
        env_name: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
        from brax import envs

        self._jax = jax
        self._jnp = jnp
        self._ravel_pytree = ravel_pytree
        self.device = torch.device(device)

        spec = ROBOT_SPECS.get(robot, {})
        env_name = env_name or spec.get("env_name", robot)
        backend = backend or spec.get("backend", "generalized")
        self.robot = robot

        self._env = envs.create(
            env_name,
            episode_length=int(episode_length),
            action_repeat=int(action_repeat),
            auto_reset=False,
            batch_size=int(n_envs),
            backend=backend,
        )
        self.num_envs = int(n_envs)
        self.n_act = int(self._env.action_size)
        self._rng = jax.random.PRNGKey(int(seed))

        # Establish the flatten/unflatten structure from an initial state.
        self._reset_jit = jax.jit(self._env.reset)
        self._rng, key = jax.random.split(self._rng)
        state0 = self._reset_jit(key)
        flat0, self._unravel = ravel_pytree(state0)
        self._flat_dim = int(flat0.shape[0])

        # JAX step over the flattened state -> (next_flat, mpc_state, task_obs, reward, done).
        def _jax_step(flat, action):
            st = self._unravel(flat)
            ns = self._env.step(st, action)
            q, qd = _q_qd(ns.pipeline_state)
            mpc_state = jnp.concatenate([q[..., -self.n_act:], qd[..., -self.n_act:]], axis=-1)
            return ravel_pytree(ns)[0], mpc_state, ns.obs, ns.reward, ns.done

        self._diff_step = jax2torch(_jax_step)

        self.mpc_state_dim = 2 * self.n_act
        self.task_obs_dim = int(state0.obs.shape[-1])
        self.obs_dim = self.mpc_state_dim + self.task_obs_dim
        self._flat_state: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def _reset_flat(self) -> torch.Tensor:
        self._rng, key = self._jax.random.split(self._rng)
        state = self._reset_jit(key)
        flat, _ = self._ravel_pytree(state)
        return torch.as_tensor(np.asarray(flat), dtype=torch.float32, device=self.device)

    def reset(self) -> torch.Tensor:
        """Reset all envs; return the carried (detached) flat state."""
        self._flat_state = self._reset_flat().requires_grad_(False)
        return self._flat_state

    def detach_state(self) -> None:
        """Truncate the gradient window (DReplay-style) without resetting physics."""
        if self._flat_state is not None:
            self._flat_state = self._flat_state.detach()

    def reset_window(self) -> None:
        """Full DReplay reset: re-init physics to the demonstration/initial state."""
        self._flat_state = self._reset_flat()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable step. ``action`` (n_envs, n_act) may carry grad.

        Returns (obs_full, reward, done, mpc_state) as torch tensors; obs_full and
        reward are differentiable w.r.t. ``action`` and the carried state.
        """
        if self._flat_state is None:
            self.reset()
        next_flat, mpc_state, task_obs, reward, done = self._diff_step(self._flat_state, action)
        self._flat_state = next_flat  # carry WITH graph for multi-step gradient
        obs_full = torch.cat([mpc_state, task_obs], dim=-1)
        return obs_full, reward, done, mpc_state

    def current_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (obs_full, mpc_state) for the current carried state (no step)."""
        flat = self._flat_state if self._flat_state is not None else self.reset()
        st = self._unravel(_t2j_local(flat))
        q, qd = _q_qd(st.pipeline_state)
        jnp = self._jnp
        mpc_state_j = jnp.concatenate([q[..., -self.n_act:], qd[..., -self.n_act:]], axis=-1)
        mpc_state = torch.as_tensor(np.asarray(mpc_state_j), dtype=torch.float32, device=self.device)
        task_obs = torch.as_tensor(np.asarray(st.obs), dtype=torch.float32, device=self.device)
        return torch.cat([mpc_state, task_obs], dim=-1), mpc_state


def _q_qd(pipeline_state: Any):
    for qa, qda in (("q", "qd"), ("qpos", "qvel")):
        if hasattr(pipeline_state, qa) and hasattr(pipeline_state, qda):
            return getattr(pipeline_state, qa), getattr(pipeline_state, qda)
    raise AttributeError("pipeline_state lacks q/qd and qpos/qvel")


def _t2j_local(t: torch.Tensor):
    import jax.numpy as jnp

    return jnp.asarray(t.detach().cpu().numpy())
