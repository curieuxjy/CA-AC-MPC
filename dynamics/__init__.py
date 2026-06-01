"""Reduced/template dynamics models for the differentiable MPC layer.

These replace the drone-specific ``DroneDx`` when porting CA-AC-MPC to other
robots (e.g. Brax locomotion). A template model only needs to expose the same
small interface the MPC consumes:

    n_state, n_ctrl, dt
    forward(x, u)        -> x_next                       (B, n_state)
    grad_input(x, u)     -> (R, S)  with R = dx_next/dx, S = dx_next/du

and a thin wrapper providing ``f_dyn(x, u, dt)`` / ``f_dyn_jac(x, u, dt)`` as
``DifferentiableMPCController`` expects.
"""

from .template_dx import (  # noqa: F401
    TemplateDoubleIntegratorDx,
    TemplateDxDiffMPCWrapper,
)
