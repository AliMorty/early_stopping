# RNG Design: Local vs Global

## Short answer

Both are fine, but they serve different goals. The current design is intentional and correct for your use case.

## What the code does now

- `generate_data`: creates `RandomState(self.seed)` locally. Same seed → same X, y every time. This is what makes resumability work (re-generate identical data without saving X, y).
- `population_logistic_loss`: creates `RandomState(999)` locally. Same MC sample every evaluation, regardless of when you call it.

## What a global `self.rng` would do differently

A single `self.rng = RandomState(seed)` shared across methods means the RNG state advances as you call methods. So:
- `generate_data()` consumes some random numbers, advancing the state.
- Later calls to `population_logistic_loss()` would get **different** samples depending on how many GD steps happened before (since each pop loss call would advance the shared state).
- Calling `population_logistic_loss` twice in a row would give **different** results each time.

## Why the current approach is better here

1. **Reproducibility of data**: `generate_data()` always produces the same X, y for the same seed, no matter what else happened before. This is critical for resumability.
2. **Stable population loss**: Every evaluation of pop loss uses the same MC sample, so the pop loss curve is smooth and comparable across checkpoints. No random jitter between evaluations.
3. **Independence**: The two methods don't accidentally interfere with each other's randomness.

## When you'd want a global rng

If you needed the population loss samples to be **independent** of the training data (truly fresh randomness each time), you'd want a separate rng. But even then, you'd want two separate rngs (`self.data_rng` and `self.pop_rng`), not one shared one.

## Bottom line

Current design: correct and intentional. Don't change it.
