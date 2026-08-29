# EquationSolver AI

Hybrid neuro-symbolic Android equation solver using an exact local solver plus a lightweight neural model.

## Current capabilities
- Linear equations in `x` or `y`
- 2x2 linear systems using `;` between equations
- Quadratic equations in `x` using `x^2` or `x²`
- Automatic equation-type detection
- Human-readable solving steps
- Local neural prediction and training with mini-batches
- Synthetic training data for linear, quadratic, and system problems
- Validation MSE reporting

The exact solver remains the ground truth for supported mathematics; the neural model learns numerical estimates and can be retrained locally.
