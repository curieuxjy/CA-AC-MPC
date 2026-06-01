"""Brax -> Stable-Baselines3 VecEnv adapter (Phase 1 of the Brax port).

This bridges a *batched, differentiable* Brax (JAX) environment into the SB3
``VecEnv`` interface that the rest of this repo (the SB3-ACMPC fork, PPO,
``VecNormalize``, ``ResetWithRawStateWrapper``) already speaks.

Design notes
------------
* The adapter is a *plain* ``VecEnv`` that returns NumPy observations with the
  layout ``[mpc_state | task_obs]`` -- the same split the drone env uses
  (``mpc_state_dim`` prefix is the raw state for the MPC, the tail is the
  policy observation). ``VecNormalize`` and ``ResetWithRawStateWrapper`` are
  layered on top by ``utils.train_support._make_vec_env_from_cfg`` exactly as
  for the gate-racing env -- so this file does not touch normalization or the
  ``(obs, state)`` reset contract.
* SB3 collects rollouts in NumPy, so the JAX<->host boundary uses
  ``np.asarray`` / ``jnp.asarray`` (a host copy on GPU; zero-cost on CPU).
  DLPack zero-copy is a later optimisation and is intentionally NOT used here.
* Brax is already massively parallel via ``jax.vmap``; we run a single process
  with ``batch_size = n_envs``. Do **not** wrap this in ``SubprocVecEnv``.
* ``make_vec_env`` (which normally adds a ``Monitor``) is bypassed, so this
  adapter emits ``info["episode"]`` itself for ``ep_rew_mean`` logging.
* We use *manual* auto-reset (jitted step + masked reset) instead of Brax's
  ``AutoResetWrapper`` so the true terminal observation is preserved for PPO
  bootstrapping (``info["terminal_observation"]`` / ``TimeLimit.truncated``).

Verification status: written against the Brax >=0.9 public API. NOT executed
in this environment (no JAX/Brax/torch installed; ``gym==0.21`` is incompatible
with the local Python 3.12). See ``tests/test_brax_vec_env.py`` for the smoke
test to run once a compatible env exists.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import gym
import numpy as np

# JAX / Brax are imported lazily inside __init__ so that merely importing this
# module (e.g. for type references) does not hard-require the JAX stack.

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

# A callback that maps a (batched) Brax State -> a JAX array.
StateFn = Callable[[Any], Any]


# --------------------------------------------------------------------------- #
# Default state/observation extractors.
# --------------------------------------------------------------------------- #
def _get_q_qd(pipeline_state: Any) -> Tuple[Any, Any]:
    """Return generalized coordinates (q, qd) from a Brax pipeline state.

    Works across backends: generalized/positional/spring expose ``.q``/``.qd``;
    the MJX backend exposes ``.qpos``/``.qvel``.
    """
    for q_attr, qd_attr in (("q", "qd"), ("qpos", "qvel")):
        if hasattr(pipeline_state, q_attr) and hasattr(pipeline_state, qd_attr):
            return getattr(pipeline_state, q_attr), getattr(pipeline_state, qd_attr)
    raise AttributeError(
        "Could not find generalized coordinates on the Brax pipeline state "
        "(tried q/qd and qpos/qvel)."
    )


def default_mpc_state_fn(state: Any):
    """Default MPC template state = concat([q, qd]) of the full system.

    Phase 2 (the reduced/template dynamics model) will refine this to only the
    actuated DOFs it models. For Phase 1 plumbing we expose the full
    generalized state; the policy never sees it directly (the obs extractor
    slices it off), it only feeds the MPC.
    """
    import jax.numpy as jnp

    q, qd = _get_q_qd(state.pipeline_state)
    return jnp.concatenate([q, qd], axis=-1)


def default_task_obs_fn(state: Any):
    """Default policy observation = Brax's native task observation."""
    return state.obs


def make_actuated_joint_state_fn(n_act: int) -> StateFn:
    """MPC state = [q_act, qd_act] for the actuated joints.

    MuJoCo-derived Brax robots order the root / free-joint DOFs first, so the
    actuated hinge joints are the *last* ``n_act`` entries of both ``q`` and
    ``qd`` (holds for halfcheetah, walker2d, hopper, ant, ...). This matches
    ``TemplateDoubleIntegratorDx`` with ``n_dof = n_act`` (n_state = 2*n_act).
    """

    def _fn(state: Any):
        import jax.numpy as jnp

        q, qd = _get_q_qd(state.pipeline_state)
        return jnp.concatenate([q[..., -n_act:], qd[..., -n_act:]], axis=-1)

    return _fn


# Minimal robot registry. ``backend`` is the Brax physics pipeline; the
# generalized pipeline is differentiable and CPU-friendly, which suits the
# no-CUDA validation phase.
ROBOT_SPECS: Dict[str, Dict[str, Any]] = {
    "halfcheetah": {"env_name": "halfcheetah", "backend": "generalized"},
    "walker2d": {"env_name": "walker2d", "backend": "generalized"},
    "ant": {"env_name": "ant", "backend": "generalized"},
}


class BraxVecEnvAdapter(VecEnv):
    """Expose a batched Brax env as an SB3 ``VecEnv``.

    Observation layout: ``[mpc_state (mpc_state_dim) | task_obs]``.
    """

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
        mpc_state_fn: StateFn = default_mpc_state_fn,
        task_obs_fn: StateFn = default_task_obs_fn,
        mpc_state_mode: str = "full",
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from brax import envs

        self._jax = jax
        self._jnp = jnp

        spec = ROBOT_SPECS.get(robot, {})
        env_name = env_name or spec.get("env_name", robot)
        backend = backend or spec.get("backend", "generalized")
        self.robot = robot
        self.env_name = env_name
        self.brax_backend = backend
        self._mpc_state_fn = mpc_state_fn
        self._task_obs_fn = task_obs_fn

        # Build a batched env WITHOUT brax auto-reset; we auto-reset manually
        # (below) so the true terminal observation survives for PPO bootstrap.
        self._env = envs.create(
            env_name,
            episode_length=int(episode_length),
            action_repeat=int(action_repeat),
            auto_reset=False,
            batch_size=int(n_envs),
            backend=backend,
            **(env_kwargs or {}),
        )

        # "actuated" mode needs the action size, which is known only after the
        # env exists -> resolve the state_fn here. "full" keeps the default.
        if str(mpc_state_mode).lower() == "actuated":
            self._mpc_state_fn = make_actuated_joint_state_fn(int(self._env.action_size))
        self.mpc_state_mode = str(mpc_state_mode).lower()

        self._rng = jax.random.PRNGKey(int(seed))
        self._reset_fn = jax.jit(self._env.reset)
        self._step_and_reset_fn = jax.jit(self._step_and_reset)

        # Probe a reset to infer dimensions and prime the current state.
        self._rng, key = jax.random.split(self._rng)
        self._state = self._reset_fn(key)
        mpc_state0 = self._mpc_state_fn(self._state)
        task_obs0 = self._task_obs_fn(self._state)
        self.mpc_state_dim = int(mpc_state0.shape[-1])
        task_obs_dim = int(task_obs0.shape[-1])
        obs_dim = self.mpc_state_dim + task_obs_dim
        act_dim = int(self._env.action_size)

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        super().__init__(int(n_envs), observation_space, action_space)

        # Episode bookkeeping (replaces SB3 Monitor, which make_vec_env would
        # otherwise add).
        self._ep_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # JAX helpers
    # ------------------------------------------------------------------ #
    def _build_obs(self, state: Any):
        jnp = self._jnp
        mpc_state = self._mpc_state_fn(state)
        task_obs = self._task_obs_fn(state)
        return jnp.concatenate([mpc_state, task_obs], axis=-1)

    def _step_and_reset(self, state: Any, action: Any, rng: Any):
        """One batched env step with masked auto-reset.

        Returns (new_state, obs_next, obs_terminal, reward, done, truncation,
        new_rng). ``obs_next`` is post-reset (the SB3 contract); ``obs_terminal``
        is the pre-reset observation for bootstrapping done envs.
        """
        jax = self._jax
        jnp = self._jnp

        stepped = self._env.step(state, action)
        obs_terminal = self._build_obs(stepped)
        reward = stepped.reward
        done = stepped.done
        truncation = stepped.info.get("truncation", jnp.zeros_like(done))

        rng, key = jax.random.split(rng)
        reset_state = self._env.reset(key)

        def _select(reset_leaf, keep_leaf):
            d = done.reshape((done.shape[0],) + (1,) * (jnp.ndim(reset_leaf) - 1))
            return jnp.where(d, reset_leaf, keep_leaf)

        new_state = jax.tree_util.tree_map(_select, reset_state, stepped)
        obs_next = self._build_obs(new_state)
        return new_state, obs_next, obs_terminal, reward, done, truncation, rng

    # ------------------------------------------------------------------ #
    # VecEnv API
    # ------------------------------------------------------------------ #
    def reset(self):
        self._rng, key = self._jax.random.split(self._rng)
        self._state = self._reset_fn(key)
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        return np.asarray(self._build_obs(self._state), dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        actions = self._jnp.asarray(self._actions)
        (
            self._state,
            obs_next,
            obs_terminal,
            reward,
            done,
            truncation,
            self._rng,
        ) = self._step_and_reset_fn(self._state, actions, self._rng)

        obs_next = np.asarray(obs_next, dtype=np.float32)
        obs_terminal = np.asarray(obs_terminal, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        done = np.asarray(done).reshape(-1).astype(bool)
        truncation = np.asarray(truncation).reshape(-1).astype(bool)

        self._ep_returns += reward
        self._ep_lengths += 1

        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if done[i]:
                infos[i]["episode"] = {
                    "r": float(self._ep_returns[i]),
                    "l": int(self._ep_lengths[i]),
                }
                infos[i]["terminal_observation"] = obs_terminal[i]
                if truncation[i]:
                    # Episode ended by the time limit, not a true terminal:
                    # PPO should bootstrap from terminal_observation.
                    infos[i]["TimeLimit.truncated"] = True
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0

        return obs_next, reward, done, infos

    def close(self) -> None:  # nothing to release (single process, JAX-managed)
        return None

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        if seed is not None:
            self._rng = self._jax.random.PRNGKey(int(seed))
        return [seed] * self.num_envs

    # --- attribute / method plumbing (minimal but spec-compliant) -------- #
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        idx = self._get_indices(indices)
        value = getattr(self, attr_name, None)
        return [value for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs
    ) -> List[Any]:
        idx = self._get_indices(indices)
        method = getattr(self, method_name, None)
        if not callable(method):
            return [None for _ in idx]
        return [method(*method_args, **method_kwargs) for _ in idx]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        idx = self._get_indices(indices)
        return [False for _ in idx]

    def get_images(self) -> Sequence[np.ndarray]:
        raise NotImplementedError("Rendering not implemented for BraxVecEnvAdapter.")
