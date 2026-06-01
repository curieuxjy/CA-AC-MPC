#!/usr/bin/env python3
"""Phase-2 unit test for the template double-integrator dynamics.

Pure PyTorch; no JAX/Brax/CUDA needed. Run once torch is installed:

    python tests/test_template_dx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamics import TemplateDoubleIntegratorDx, TemplateDxDiffMPCWrapper  # noqa: E402


def _finite_diff_jac(f, x, u, eps=1e-4):
    B, nx = x.shape
    nu = u.shape[1]
    R = torch.zeros(B, nx, nx)
    S = torch.zeros(B, nx, nu)
    for j in range(nx):
        d = torch.zeros_like(x)
        d[:, j] = eps
        R[:, :, j] = (f(x + d, u) - f(x - d, u)) / (2 * eps)
    for j in range(nu):
        d = torch.zeros_like(u)
        d[:, j] = eps
        S[:, :, j] = (f(x, u + d) - f(x, u - d)) / (2 * eps)
    return R, S


def test_shapes_and_jacobian():
    torch.manual_seed(0)
    n_dof, dt, B = 6, 0.05, 8
    dx = TemplateDoubleIntegratorDx(n_dof, dt=dt, ctrl_scale=2.0)
    assert dx.n_state == 12 and dx.n_ctrl == 6

    x = torch.randn(B, dx.n_state)
    u = torch.randn(B, dx.n_ctrl)

    x_next = dx.forward(x, u)
    assert x_next.shape == (B, dx.n_state)

    R, S = dx.grad_input(x, u)
    assert R.shape == (B, dx.n_state, dx.n_state)
    assert S.shape == (B, dx.n_state, dx.n_ctrl)

    R_fd, S_fd = _finite_diff_jac(dx.forward, x, u)
    assert torch.allclose(R, R_fd, atol=1e-4), (R - R_fd).abs().max()
    assert torch.allclose(S, S_fd, atol=1e-4), (S - S_fd).abs().max()
    print("[ok] shapes + analytic Jacobian matches finite-difference")


def test_linear_exactness():
    """For a linear system the Jacobian reproduces forward exactly."""
    torch.manual_seed(1)
    dx = TemplateDoubleIntegratorDx(4, dt=0.02, ctrl_scale=1.5)
    x = torch.randn(5, dx.n_state)
    u = torch.randn(5, dx.n_ctrl)
    R, S = dx.grad_input(x, u)
    base = dx.forward(x, u)
    ddx = 0.1 * torch.randn_like(x)
    ddu = 0.1 * torch.randn_like(u)
    pred = base + torch.bmm(R, ddx.unsqueeze(-1)).squeeze(-1) + torch.bmm(S, ddu.unsqueeze(-1)).squeeze(-1)
    actual = dx.forward(x + ddx, u + ddu)
    assert torch.allclose(pred, actual, atol=1e-5), (pred - actual).abs().max()
    print("[ok] linear-exactness: forward == base + R·dx + S·du")


def test_wrapper_interface():
    dx = TemplateDoubleIntegratorDx(3, dt=0.05)
    w = TemplateDxDiffMPCWrapper(dx)
    x = torch.randn(2, dx.n_state)
    u = torch.randn(2, dx.n_ctrl)
    assert torch.allclose(w.f_dyn(x, u, dx.dt), dx.forward(x, u))
    R, S = w.f_dyn_jac(x, u, dx.dt)
    assert R.shape == (2, dx.n_state, dx.n_state)
    print("[ok] DiffMPC wrapper interface")


def test_custom_B_a():
    """Non-square B_a (n_ctrl != n_dof) is supported."""
    B_a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 dof, 2 ctrl
    dx = TemplateDoubleIntegratorDx(3, dt=0.05, B_a=B_a)
    assert dx.n_ctrl == 2 and dx.n_state == 6
    x = torch.randn(4, 6)
    u = torch.randn(4, 2)
    R, S = dx.grad_input(x, u)
    R_fd, S_fd = _finite_diff_jac(dx.forward, x, u)
    assert torch.allclose(S, S_fd, atol=1e-4)
    print("[ok] custom non-square B_a")


if __name__ == "__main__":
    test_shapes_and_jacobian()
    test_linear_exactness()
    test_wrapper_interface()
    test_custom_B_a()
    print("\nAll Phase-2 template-dynamics tests passed.")
