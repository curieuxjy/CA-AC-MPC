"""Generic locomotion env from an arbitrary MJCF, MJX-ready (complex-robot port).

Brax ships only a few simple built-in robots. This loads *any* MJCF file
(`brax.io.mjcf.load`) and exposes a standard locomotion task (forward velocity
+ healthy bonus - control cost), so the CA-AC-MPC Brax port can be run on
complex robots (e.g. a humanoid from MuJoCo Menagerie) without writing a new
env per robot.

Modeled on Brax's own humanoid env (same reset/step/reward structure, same MJX
solver config) but generalized: the MJCF path, torso body index, healthy
z-range and reward weights are all parameters.

URDF: convert to MJCF first (MuJoCo can compile URDF; Brax's native URDF loader
is legacy). See docs/SETUP_CUDA.md.

EXPERIMENTAL: not executed here (no mujoco/mjx/brax installed). Assumes a
floating-base (freejoint) locomotion robot: the first 7 qpos / 6 qvel are the
free base, so qpos[2:] keeps z+orientation+joints. Verify torso_index / joint
layout for your specific MJCF.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
from jax import numpy as jp

import brax.envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


def _resolve_mjcf_path(mjcf_path: Optional[str]) -> str:
    """None / 'brax_humanoid' -> Brax's bundled (MJX-ready) humanoid.xml.

    This gives a guaranteed-working default; swap ``mjcf_path`` for any custom
    MJCF (e.g. a Menagerie humanoid) once the pipeline is verified.
    """
    if mjcf_path in (None, "", "brax_humanoid", "brax:humanoid"):
        from etils import epath

        return (epath.resource_path("brax") / "envs/assets/humanoid.xml").as_posix()
    return str(mjcf_path)


class MjcfLocomotionEnv(PipelineEnv):
    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        backend: str = "mjx",
        n_frames: int = 5,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        torso_index: int = 0,
        **kwargs,
    ):
        sys = mjcf.load(_resolve_mjcf_path(mjcf_path))

        if backend == "mjx":
            import mujoco

            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        elif backend in ("spring", "positional"):
            sys = sys.tree_replace({"opt.timestep": 0.0015})
            n_frames = 10

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        self._forward_reward_weight = float(forward_reward_weight)
        self._ctrl_cost_weight = float(ctrl_cost_weight)
        self._healthy_reward = float(healthy_reward)
        self._terminate_when_unhealthy = bool(terminate_when_unhealthy)
        self._healthy_z_range = tuple(healthy_z_range)
        self._reset_noise_scale = float(reset_noise_scale)
        self._exclude_current_positions = bool(exclude_current_positions_from_observation)
        self._torso = int(torso_index)

    def reset(self, rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        lo, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(r1, (self.sys.q_size(),), minval=lo, maxval=hi)
        qvel = jax.random.uniform(r2, (self.sys.qd_size(),), minval=lo, maxval=hi)
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        metrics = {"forward_velocity": jp.zeros(()), "is_healthy": jp.zeros(()), "ctrl_cost": jp.zeros(())}
        return State(pipeline_state, obs, jp.zeros(()), jp.zeros(()), metrics)

    def step(self, state, action):
        ps0 = state.pipeline_state
        ps = self.pipeline_step(ps0, action)

        x_before = ps0.x.pos[self._torso, 0]
        x_after = ps.x.pos[self._torso, 0]
        velocity = (x_after - x_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity

        z = ps.x.pos[self._torso, 2]
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(z < min_z, 0.0, 1.0)
        is_healthy = jp.where(z > max_z, 0.0, is_healthy)
        healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        reward = forward_reward + healthy_reward - ctrl_cost
        done = (1.0 - is_healthy) if self._terminate_when_unhealthy else jp.zeros(())

        obs = self._get_obs(ps)
        metrics = {"forward_velocity": velocity, "is_healthy": is_healthy, "ctrl_cost": ctrl_cost}
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs(self, pipeline_state):
        qpos = pipeline_state.q
        qvel = pipeline_state.qd
        if self._exclude_current_positions:
            qpos = qpos[2:]  # drop free-base x, y (translation-invariant locomotion obs)
        return jp.concatenate([qpos, qvel])


# Register so envs.create('mjcf_locomotion', mjcf_path=..., backend='mjx', ...) works
# and reuses the standard episode/vmap wrappers (same path as the built-in robots).
brax_envs.register_environment("mjcf_locomotion", MjcfLocomotionEnv)
