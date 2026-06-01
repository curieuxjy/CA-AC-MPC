"""Actor-Critic MPC policy for Brax robots (Phase 3 of the Brax port).

This is the dimension-generalised sibling of ``MlpMpcPolicyDiffMPC``: instead of
the drone-specific ``DroneDx`` (nx=10, nu=4) and its 14-block cost layout, it
drives the reduced ``TemplateDoubleIntegratorDx`` (nx=2*n_act, nu=n_act) and a
generic diagonal-Q + linear-c cost over ``n_tau = nx + nu`` dimensions.

Kept intentionally separate from ``mlp_mpc_policy_diffmpc.py`` so the validated
drone path is untouched. Differences from the drone policy:
  * dynamics: TemplateDoubleIntegratorDx (analytic constant Jacobians);
  * backend: ``diffmpc`` only (the ``fast`` CUDA kernels are drone-specific);
  * cost: generic per-dimension diagonal Q (>0) and linear c, no p/q/v/w/t split;
  * action: the MPC's first control u_0 IS the (identity-distribution) action
    mean, mapped to the [-1, 1] Brax action box by ``u / u_max`` -- no thrust /
    body-rate normalisation.

The actor's job is to shape the cost so the *template*-model MPC produces
controls that work in the *true* Brax dynamics (the model-mismatch the AC-MPC
premise relies on).

Tunable via env vars (set by train_acmpc from config):
  ACMPC_T                MPC horizon
  ACMPC_MPC_MAX_ITER     iLQR iterations
  ACMPC_BRAX_DT          template step size (s)            default 0.05
  ACMPC_BRAX_CTRL_SCALE  accel per unit action             default 1.0
  ACMPC_BRAX_RANGE_Q     max diagonal Q weight             default 100.0
  ACMPC_BRAX_RANGE_P     half-range of the linear c term   default 10.0
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Tuple

import torch as th
from torch import nn

_HERE = Path(__file__).resolve()
_ACMPC_ROOT = _HERE.parents[1]
_REPO_ROOT = _ACMPC_ROOT.parent
sys.path.insert(0, str(_ACMPC_ROOT / "mpc.pytorch"))
sys.path.insert(0, str(_REPO_ROOT))

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod  # noqa: E402
from dynamics import TemplateDoubleIntegratorDx, TemplateDxDiffMPCWrapper  # noqa: E402

from gym import spaces  # noqa: E402

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
except ModuleNotFoundError:
    class ActorCriticPolicy(nn.Module):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "stable_baselines3 is required to instantiate MlpMpcPolicyBrax."
            )


class CustomNetworkBrax(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_ctrl: int,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()
        self.features_in_dim = int(feature_dim)
        self.T = int(os.environ["ACMPC_T"])

        n_act = int(n_ctrl)
        self.dt = float(os.environ.get("ACMPC_BRAX_DT", "0.05"))
        ctrl_scale = float(os.environ.get("ACMPC_BRAX_CTRL_SCALE", "1.0"))
        self.range_q = float(os.environ.get("ACMPC_BRAX_RANGE_Q", "100.0"))
        self.range_p = float(os.environ.get("ACMPC_BRAX_RANGE_P", "10.0"))

        self.mpc_device = self._resolve_mpc_device()
        self.mpc_backend = "diffmpc"  # fast (CUDA) is drone-specific
        self.mpc_grad_mode = self._resolve_mpc_grad_mode()
        self.predictions = None
        self.last_cost_vectors = None

        self.dx = TemplateDoubleIntegratorDx(
            n_dof=n_act, dt=self.dt, ctrl_scale=ctrl_scale, device=str(self.mpc_device)
        )
        self.dx_wrapper = TemplateDxDiffMPCWrapper(self.dx)

        n_tau = self.dx.n_state + self.dx.n_ctrl
        self.n_output = 2 * n_tau * self.T  # diagonal Q (n_tau) + linear c (n_tau), per step

        self.latent_dim_pi = n_act
        self.latent_dim_vf = int(last_layer_dim_vf)

        # Same Sequential shape as the drone policy so index [6] is the last
        # Linear (orthogonal-init hook in MlpMpcPolicyBrax relies on this).
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, self.n_output), nn.Sigmoid(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, self.latent_dim_vf), nn.GELU(),
        )

        self.mpc_controller = self._build_mpc_controller()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_mpc_device() -> th.device:
        preferred = os.environ.get("ACMPC_MPC_DEVICE")
        if preferred is None:
            preferred = "cuda:0" if th.cuda.is_available() else "cpu"
        if preferred.startswith("cuda") and not th.cuda.is_available():
            return th.device("cpu")
        return th.device(preferred)

    @staticmethod
    def _resolve_mpc_grad_mode() -> str:
        mode = os.environ.get("ACMPC_MPC_GRAD_MODE", "diff").strip().lower()
        if mode not in {"diff", "stop", "unroll"}:
            raise ValueError(f"Unsupported ACMPC_MPC_GRAD_MODE={mode}.")
        return mode

    def _build_mpc_controller(self) -> DifferentiableMPCController:
        nx, nu = self.dx.n_state, self.dx.n_ctrl
        n_tau = nx + nu
        C0 = th.zeros(1, self.T, n_tau, n_tau, device=self.mpc_device)
        c0 = th.zeros(1, self.T, n_tau, device=self.mpc_device)
        C0_final = th.zeros(1, n_tau, n_tau, device=self.mpc_device)
        c0_final = th.zeros(1, n_tau, device=self.mpc_device)
        cost_module = GeneralQuadCost(
            nx=nx, nu=nu, C=C0, c=c0, C_final=C0_final, c_final=c0_final,
            device=str(self.mpc_device),
        )
        u_min = self.dx.u_min.to(self.mpc_device)
        u_max = self.dx.u_max.to(self.mpc_device)
        max_iter = int(os.environ.get("ACMPC_MPC_MAX_ITER", "5"))
        return DifferentiableMPCController(
            f_dyn=self.dx_wrapper.f_dyn,
            f_dyn_jac=self.dx_wrapper.f_dyn_jac,
            total_time=self.T * self.dx.dt,
            step_size=self.dx.dt,
            horizon=self.T,
            cost_module=cost_module,
            u_min=u_min,
            u_max=u_max,
            grad_method=GradMethod.ANALYTIC,
            max_iter=max_iter,
            exit_unconverged=False,
            detach_unconverged=False,
            device=str(self.mpc_device),
            verbose=0,
            use_vmap_line_search=False,
        )

    def _build_cost_tensors(
        self, sigmoid_cost: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        batch_size = sigmoid_cost.shape[0]
        horizon = self.T
        n_tau = self.dx.n_state + self.dx.n_ctrl
        epsilon = 0.01
        half = n_tau * horizon

        # First half -> strictly-positive diagonal Q weights (state + ctrl).
        q_diag = sigmoid_cost[:, :half] * self.range_q + epsilon
        # Second half -> signed linear term (acts like a learned setpoint bias).
        c_lin = (sigmoid_cost[:, half:] - 0.5) * self.range_p

        C = th.zeros(batch_size, horizon, n_tau, n_tau, device=self.mpc_device)
        c = th.zeros(batch_size, horizon, n_tau, device=self.mpc_device)
        for t in range(horizon):
            sl = slice(t * n_tau, (t + 1) * n_tau)
            C[:, t] = th.diag_embed(q_diag[:, sl])
            c[:, t] = c_lin[:, sl]

        C_final = th.zeros(batch_size, n_tau, n_tau, device=self.mpc_device)
        c_final = th.zeros(batch_size, n_tau, device=self.mpc_device)
        return C, c, C_final, c_final

    def _solve_chunk_diffmpc(self, states_chunk, C, c, C_final, c_final, U_init):
        n_chunk = states_chunk.shape[0]
        x_ref = th.zeros(n_chunk, self.T + 1, self.dx.n_state, device=self.mpc_device)
        u_ref = th.zeros(n_chunk, self.T, self.dx.n_ctrl, device=self.mpc_device)
        self.mpc_controller.cost_module.C = C
        self.mpc_controller.cost_module.c = c
        self.mpc_controller.cost_module.C_final = C_final
        self.mpc_controller.cost_module.c_final = c_final
        self.mpc_controller.cost_module.set_reference(x_ref, u_ref)
        self.mpc_controller.preserve_gradients = bool(
            self.mpc_grad_mode == "unroll" and th.is_grad_enabled()
        )
        return self.mpc_controller(states_chunk, U_init=U_init)

    # ------------------------------------------------------------------ #
    def forward(self, features: th.Tensor, states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features, states), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, states: th.Tensor) -> th.Tensor:
        policy_device = features.device
        states = states.float()
        if states.ndimension() == 1:
            states = states.unsqueeze(0)
        states = states[:, : self.dx.n_state]

        features_in = features[:, : self.features_in_dim]
        sigmoid_cost_all = self.policy_net(features_in)
        self.last_cost_vectors = sigmoid_cost_all.detach().to("cpu")

        n_batch = features.shape[0]
        chunk_length = int(os.environ.get("ACMPC_MPC_CHUNK", "1024"))
        nom_x = th.zeros((n_batch, self.T, self.dx.n_state), device=self.mpc_device)
        nom_u = th.zeros((n_batch, self.T, self.dx.n_ctrl), device=self.mpc_device)

        idx_start = 0
        for sigmoid_cost in th.split(sigmoid_cost_all, chunk_length, dim=0):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk
            states_chunk = states[idx_start:idx_end].to(self.mpc_device)
            C, c, C_final, c_final = self._build_cost_tensors(sigmoid_cost.to(self.mpc_device))
            U_init = th.zeros(n_chunk, self.T, self.dx.n_ctrl, device=self.mpc_device)

            solve_ctx = (
                th.no_grad()
                if (self.mpc_grad_mode == "stop" and th.is_grad_enabled())
                else nullcontext()
            )
            with solve_ctx:
                X_chunk, U_chunk = self._solve_chunk_diffmpc(
                    states_chunk, C, c, C_final, c_final, U_init
                )
            if self.mpc_grad_mode == "stop" and th.is_grad_enabled():
                X_chunk, U_chunk = X_chunk.detach(), U_chunk.detach()

            nom_x[idx_start:idx_end] = X_chunk[:, : self.T, :]
            nom_u[idx_start:idx_end] = U_chunk
            idx_start = idx_end

        self.predictions = th.cat((nom_x, nom_u), dim=2).detach()

        # First control = identity-distribution action mean, mapped to [-1, 1].
        u0 = nom_u[:, 0, :]
        u_max = self.dx.u_max.to(device=self.mpc_device, dtype=u0.dtype)
        action = u0 / u_max
        return action.to(policy_device)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, : self.features_in_dim]
        return self.value_net(features_in)


class MlpMpcPolicyBrax(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            distr_identity=True,
            *args,
            **kwargs,
        )
        last_linear_layer = self.mlp_extractor.policy_net[6]
        nn.init.orthogonal_(last_linear_layer.weight, gain=0.5)
        nn.init.constant_(last_linear_layer.bias, 0.0)

    def _build_mlp_extractor(self) -> None:
        n_ctrl = int(self.action_space.shape[0])
        self.mlp_extractor = CustomNetworkBrax(self.features_dim, n_ctrl=n_ctrl)
