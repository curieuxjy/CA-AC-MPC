#!/usr/bin/env python3
"""Phase-7 test: jax2torch gradient correctness.

Checks that gradients computed through the jax2torch autograd Function match
JAX's own ``jax.grad`` for a non-trivial function. Needs jax + torch (CPU OK):

    python tests/test_jax_torch_bridge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.jax_torch_bridge import jax2torch  # noqa: E402


def test_scalar_grad_matches_jax():
    import jax
    import jax.numpy as jnp

    def f(x, y):
        return jnp.sum(jnp.sin(x) * y**2)

    tf = jax2torch(f)

    xt = torch.randn(5, dtype=torch.float64, requires_grad=True)
    yt = torch.randn(5, dtype=torch.float64, requires_grad=True)
    out = tf(xt, yt)
    out.backward()

    gx_jax, gy_jax = jax.grad(f, argnums=(0, 1))(
        jnp.asarray(xt.detach().numpy()), jnp.asarray(yt.detach().numpy())
    )
    import numpy as np

    assert np.allclose(xt.grad.numpy(), np.asarray(gx_jax), atol=1e-6), "x grad mismatch"
    assert np.allclose(yt.grad.numpy(), np.asarray(gy_jax), atol=1e-6), "y grad mismatch"
    print("[ok] jax2torch scalar grad matches jax.grad")


def test_multi_output_and_partial_use():
    """Only one of two outputs feeds the loss; the unused output's cotangent
    must be handled (zero) without error."""
    import jax.numpy as jnp

    def f(x):
        return jnp.sum(x**2), jnp.cos(x)  # (scalar, vector)

    tf = jax2torch(f)
    xt = torch.randn(4, dtype=torch.float64, requires_grad=True)
    s, _vec = tf(xt)
    s.backward()
    assert xt.grad is not None and torch.isfinite(xt.grad).all()
    assert torch.allclose(xt.grad, 2 * xt.detach(), atol=1e-6)
    print("[ok] jax2torch multi-output + partial-use backward")


if __name__ == "__main__":
    test_scalar_grad_matches_jax()
    test_multi_output_and_partial_use()
    print("\nAll Phase-7 jax2torch bridge tests passed.")
