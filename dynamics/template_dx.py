"""Template (reduced) dynamics for multi-DOF robots: a per-DOF double integrator.

This is the drop-in replacement for ``DroneDx`` chosen for the Brax port
(see ``docs/brax_port_plan.md``, Phase 2). The model treats the MPC state as

    x = [q, qd]      (q: n_dof positions, qd: n_dof velocities)  -> n_state = 2*n_dof

and the control as a commanded acceleration mapped through ``B_a`` (n_dof x n_ctrl):

    q_ddot = ctrl_scale * (B_a @ u)

Discretised with forward Euler (matching DroneDx.grad_input's R = I + dt*A,
S = dt*B convention):

    q_next  = q  + dt * qd
    qd_next = qd + dt * (ctrl_scale * B_a @ u)

Because the system is *linear and time-invariant*, the discrete Jacobians are
**constant**:

    R = [[I, dt*I], [0, I]]                  (n_state x n_state)
    S = [[0], [dt * ctrl_scale * B_a]]       (n_state x n_ctrl)

so ``forward`` and ``grad_input`` are *exactly* consistent (no linearisation
error) -- the iLQR linearisation reproduces ``forward`` precisely. The gap
between this template and the real Brax dynamics is the intended model mismatch
that the actor's learned cost absorbs (the AC-MPC premise).

Pure PyTorch; no JAX/Brax dependency. Runnable/testable on CPU without CUDA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class TemplateDoubleIntegratorDx(nn.Module):
    def __init__(
        self,
        n_dof: int,
        *,
        dt: float = 0.05,
        n_ctrl: Optional[int] = None,
        B_a: Optional[Tensor] = None,
        ctrl_scale: float = 1.0,
        u_min: Optional[Tensor] = None,
        u_max: Optional[Tensor] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dt = float(dt)
        self.n_dof = int(n_dof)
        self.n_state = 2 * int(n_dof)
        self.ctrl_scale = float(ctrl_scale)

        if B_a is None:
            nu = int(n_ctrl) if n_ctrl is not None else int(n_dof)
            B_a = torch.eye(self.n_dof, nu, device=self.device)
        else:
            B_a = torch.as_tensor(B_a, dtype=torch.float32, device=self.device)
            if B_a.shape[0] != self.n_dof:
                raise ValueError(f"B_a rows ({B_a.shape[0]}) must equal n_dof ({self.n_dof}).")
        self.n_ctrl = int(B_a.shape[1])
        self.register_buffer("B_a", B_a)

        # Constant discrete-time Jacobian templates (1, n_state, *) for batch repeat.
        eye_d = torch.eye(self.n_dof, device=self.device)
        R0 = torch.zeros(self.n_state, self.n_state, device=self.device)
        R0[: self.n_dof, : self.n_dof] = eye_d
        R0[: self.n_dof, self.n_dof :] = self.dt * eye_d
        R0[self.n_dof :, self.n_dof :] = eye_d
        self.register_buffer("_R0", R0.unsqueeze(0))

        S0 = torch.zeros(self.n_state, self.n_ctrl, device=self.device)
        S0[self.n_dof :, :] = self.dt * self.ctrl_scale * B_a
        self.register_buffer("_S0", S0.unsqueeze(0))

        # Control bounds (default symmetric unit box, matching Brax [-1, 1] actions).
        if u_min is None:
            u_min = -torch.ones(self.n_ctrl, device=self.device)
        if u_max is None:
            u_max = torch.ones(self.n_ctrl, device=self.device)
        self.register_buffer("u_min", torch.as_tensor(u_min, dtype=torch.float32, device=self.device))
        self.register_buffer("u_max", torch.as_tensor(u_max, dtype=torch.float32, device=self.device))

        # Generic iLQR/solver knobs the policy/controller may read (defaults that
        # work for a well-conditioned linear model). Kept attribute-compatible
        # with how the policy queries DroneDx.
        self.mass = 1.0
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 10

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        squeeze = x.ndimension() == 1
        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.ndimension() == 2 and u.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == self.n_state, f"expected n_state={self.n_state}, got {x.shape[1]}"
        assert u.shape[1] == self.n_ctrl, f"expected n_ctrl={self.n_ctrl}, got {u.shape[1]}"

        q = x[:, : self.n_dof]
        qd = x[:, self.n_dof :]
        # a = ctrl_scale * (B_a @ u): u (B, n_ctrl) @ B_a^T (n_ctrl, n_dof) -> (B, n_dof)
        a = self.ctrl_scale * torch.matmul(u, self.B_a.t())
        q_next = q + self.dt * qd
        qd_next = qd + self.dt * a
        x_next = torch.cat([q_next, qd_next], dim=1)
        return x_next.squeeze(0) if squeeze else x_next

    def grad_input(self, x: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        n_grad = x.shape[0]
        R = self._R0.expand(n_grad, self.n_state, self.n_state)
        S = self._S0.expand(n_grad, self.n_state, self.n_ctrl)
        return R, S


class TemplateDxDiffMPCWrapper:
    """Adapter exposing the (f_dyn, f_dyn_jac) signature the MPC controller wants.

    Mirrors ``DroneDxDiffMPCWrapper`` so the policy can swap dynamics models
    without changing the controller construction.
    """

    def __init__(self, dx_model: TemplateDoubleIntegratorDx):
        self.dx_model = dx_model
        self.dt = dx_model.dt

    def f_dyn(self, x: Tensor, u: Tensor, _dt: float) -> Tensor:
        return self.dx_model.forward(x, u)

    def f_dyn_jac(self, x: Tensor, u: Tensor, _dt: float) -> Tuple[Tensor, Tensor]:
        return self.dx_model.grad_input(x, u)
