#!/usr/bin/env python3
"""Phase-5 integration test: template dynamics <-> differentiable iLQR <-> cost.

Validates the riskiest new coupling introduced by the Brax port -- the
``TemplateDoubleIntegratorDx`` driving ``DifferentiableMPCController`` with an
actor-style quadratic cost, and gradients flowing from the action back to the
cost (the parameters the actor predicts). Needs only torch + DifferentialMPC;
no JAX/Brax/CUDA. Run:

    python tests/test_template_mpc_solve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "mpc.pytorch"))

from DifferentialMPC import DifferentiableMPCController, GeneralQuadCost, GradMethod  # noqa: E402
from dynamics import TemplateDoubleIntegratorDx, TemplateDxDiffMPCWrapper  # noqa: E402


def _build(T=3, n_dof=6, B=4, dt=0.05, max_iter=5):
    dx = TemplateDoubleIntegratorDx(n_dof, dt=dt)
    wrapper = TemplateDxDiffMPCWrapper(dx)
    nx, nu = dx.n_state, dx.n_ctrl
    n_tau = nx + nu

    # Actor-style cost: strictly-positive diagonal Q + signed linear c, both leaves.
    q_diag = (torch.rand(B, T, n_tau) * 10.0 + 0.1).requires_grad_(True)
    c_lin = (torch.randn(B, T, n_tau) * 0.1).requires_grad_(True)
    C = torch.diag_embed(q_diag)
    C_final = torch.zeros(B, n_tau, n_tau)
    c_final = torch.zeros(B, n_tau)

    cost = GeneralQuadCost(nx, nu, C, c_lin, C_final, c_final, device="cpu")
    cost.set_reference(torch.zeros(B, T + 1, nx), torch.zeros(B, T, nu))

    ctrl = DifferentiableMPCController(
        f_dyn=wrapper.f_dyn,
        f_dyn_jac=wrapper.f_dyn_jac,
        total_time=T * dt,
        step_size=dt,
        horizon=T,
        cost_module=cost,
        u_min=dx.u_min,
        u_max=dx.u_max,
        grad_method=GradMethod.ANALYTIC,
        max_iter=max_iter,
        exit_unconverged=False,
        detach_unconverged=False,
        device="cpu",
        verbose=0,
        use_vmap_line_search=False,
    )
    return dx, ctrl, q_diag, c_lin, nx, nu, B, T


def test_solve_finite_and_bounded():
    dx, ctrl, _, _, nx, nu, B, T = _build()
    x0 = torch.randn(B, nx)
    X, U = ctrl(x0, U_init=torch.zeros(B, T, nu))
    assert U.shape[0] == B and U.shape[-1] == nu
    assert torch.isfinite(X).all() and torch.isfinite(U).all()
    assert (U <= dx.u_max + 1e-3).all() and (U >= dx.u_min - 1e-3).all()
    print(f"[ok] solve: X{tuple(X.shape)} U{tuple(U.shape)} finite, bounds respected")


def test_gradient_flows_to_cost():
    """d(action)/d(cost params) must exist and be finite -- this is what lets
    PPO's gradient reach the actor through the MPC layer."""
    dx, ctrl, q_diag, c_lin, nx, nu, B, T = _build()
    x0 = torch.randn(B, nx)
    _, U = ctrl(x0, U_init=torch.zeros(B, T, nu))
    loss = (U[:, 0, :] ** 2).sum()
    loss.backward()
    assert c_lin.grad is not None and torch.isfinite(c_lin.grad).all()
    assert c_lin.grad.abs().sum() > 0, "no gradient reached the linear cost term"
    print("[ok] gradient flows: action -> MPC -> cost params (d u0 / d c != 0)")


if __name__ == "__main__":
    test_solve_finite_and_bounded()
    test_gradient_flows_to_cost()
    print("\nAll Phase-5 template-MPC integration tests passed.")
