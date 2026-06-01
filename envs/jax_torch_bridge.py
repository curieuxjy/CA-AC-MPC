"""jax2torch: make a JAX function differentiable inside PyTorch (Phase 7 / APG).

Wraps a pure JAX function ``f(*arrays) -> array | tuple[array]`` as a
``torch.autograd.Function`` whose forward runs ``f`` and whose backward runs
``jax.vjp``. This is what lets an Analytic Policy Gradient (APG) flow through
the *differentiable Brax simulator* into a PyTorch policy.

EXPERIMENTAL. The JAX<->PyTorch boundary here uses DLPack (zero-copy when the
devices match) and is the fragile part of the PyTorch+bridge strategy. If this
proves brittle in practice, the cleaner route is a JAX-native rewrite of the
actor-MPC (see docs/brax_port_plan.md, Phase 7 notes).

Tensors are moved by DLPack; the JAX side must run on the same device as the
torch tensors for zero-copy (otherwise a host copy happens).
"""

from __future__ import annotations

from typing import Callable

import torch


def _t2j(t: "torch.Tensor"):
    import jax.numpy as jnp

    t = t.detach().contiguous()  # grad is handled by the autograd.Function, not inside JAX
    try:
        return jnp.from_dlpack(t)
    except Exception:
        import numpy as np

        return jnp.asarray(t.cpu().numpy())


def _j2t(x) -> "torch.Tensor":
    try:
        return torch.utils.dlpack.from_dlpack(x.__dlpack__())
    except Exception:
        import numpy as np

        return torch.from_numpy(np.asarray(x))


def jax2torch(jax_fn: Callable) -> Callable:
    """Return a torch-callable wrapping ``jax_fn`` with autograd support.

    ``jax_fn`` takes one or more JAX arrays and returns a single array or a
    tuple of arrays. The returned callable takes/returns torch tensors and is
    differentiable w.r.t. any input that ``requires_grad``.
    """
    import jax

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *torch_args):
            jargs = tuple(_t2j(a) for a in torch_args)
            ys, vjp_fn = jax.vjp(jax_fn, *jargs)
            ctx.vjp_fn = vjp_fn
            ctx.single = not isinstance(ys, tuple)
            ys_t = (ys,) if ctx.single else tuple(ys)
            # Remember output specs to synthesise zero cotangents for unused outputs.
            ctx.out_specs = [(y.shape, y.dtype) for y in ys_t]
            return tuple(_j2t(y) for y in ys_t)

        @staticmethod
        def backward(ctx, *grad_outputs):
            import jax.numpy as jnp

            cot = []
            for g, (shape, dtype) in zip(grad_outputs, ctx.out_specs):
                if g is None:
                    cot.append(jnp.zeros(shape, dtype=dtype))
                else:
                    cot.append(_t2j(g))
            cot_in = cot[0] if ctx.single else tuple(cot)
            grads_in = ctx.vjp_fn(cot_in)
            return tuple(_j2t(g) for g in grads_in)

    def wrapped(*torch_args):
        out = _Fn.apply(*torch_args)
        return out[0] if len(out) == 1 else out

    return wrapped
