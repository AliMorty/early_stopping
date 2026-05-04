# Multi-Trajectory RNG Design

## The setup

You want M runs with the same (n, d, k, w*, eigenvalues, eta) but different random data. Then average the metrics and plot confidence bands.

## What needs to differ across runs

Only one thing: the training data (X, y). Each run should see independent data drawn from the same distribution.

## What should stay the same across runs

- w*, eigenvalues, eta, d, n, k (the config)
- Population loss MC samples (they approximate the same expectation — using the same MC seed across runs reduces noise in the average)
- w_tilde depends on (X, y), so it will naturally differ per run

## How to handle seeds

Use the existing `seed` parameter. Run m=0,1,...,M-1 with `seed=m` (or any M distinct seeds). Each seed produces a different `generate_data()` call → different X, y → different trajectory.

No code changes needed for the RNG. The current design already supports this:

```python
for seed in range(M):
    model = create_model(n, d, k, seed=seed)
    model.generate_data()       # different X, y each time
    model.run_gd(T, track_population_loss=True)
    model.compute_max_margin_direction()
    run_and_save(model)         # saves as run_n{}_d{}_k{}_seed{}.pkl
```

Each pkl is independent. Dashboard 3 loads all pkls for the same (n, d, k), computes metrics for each, then averages.

## Population loss seed (999)

Keep it fixed at 999 across all runs. This means every run evaluates pop loss on the same MC sample, which is what you want — it removes MC noise from the confidence bands so they only reflect data randomness, not evaluation noise.

## Summary

| Thing | Varies across runs? | Controlled by |
|-------|---------------------|---------------|
| Training data (X, y) | Yes | `seed` param (0, 1, ..., M-1) |
| Pop loss MC samples | No | hardcoded seed=999 |
| Config (n,d,k,eta,w*) | No | same for all M runs |
| w_tilde | Yes (derived from X,y) | follows from seed |
