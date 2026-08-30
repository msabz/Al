# EquationSolver AI v3

On-device Android project for training a neural model to predict solutions of mathematical equations. Geometry is intentionally outside the training curriculum.

## Neural model

The prediction shown as **جواب النموذج** comes only from the neural network. The ground-truth solver is not called by `ModelManager.predict()`.

Architecture tuned for sustained training on a mid-range Android phone:

- Positional mathematical tokenizer, up to 72 tokens
- Learned 24-dimensional token embeddings
- Dense hidden layers: `128 -> 128 -> 64`
- Numeric output: `[x, y]`
- Adam optimizer
- Mini-batch size: 24
- Binary checkpoints include weights, embeddings, Adam moments, and optimizer step

## Continuous curriculum

Synthetic examples are generated indefinitely until the user stops training. Current families include:

- Linear equations in `x` and `y`
- 2x2 linear systems
- Quadratic equations
- Cubic equations
- Quartic equations
- Rational equations
- Radical/square-root equations
- Exponential equations using both `exp(...)` and `a^x`
- Natural-log and base-10 logarithmic equations (`ln`, `log`)
- Absolute-value equations
- Trigonometric equations using `sin`, `cos`, and `tan`

Every generated example carries its known target and is independently substituted back into the equation before it is allowed into a training batch. This prevents a solver bug from silently poisoning the neural training labels.

## Test screen

The test screen deliberately shows two separate results:

1. **الجواب الصحيح** — exact or numerical teacher result used only as a reference.
2. **جواب النموذج** — neural prediction from the saved model.

For equations with several real roots, the current fixed numeric model learns a deterministic principal root: the real root closest to zero, with the lower numeric root used to break equal-distance ties. The reference text may still display several valid roots.

## Background training

Training runs in a foreground service with a partial wake lock, periodic recoverable checkpoints, battery protection, and thermal pausing. It can continue when leaving the activity or turning the screen off. Android `Force stop` still stops the app by operating-system design.

## CI

Pull requests must run unit tests before the debug APK is built. Regression tests cover linear systems, `y` handling, quadratic principal-root selection, general numerical principal roots, the expression evaluator, and generated curriculum validity.
