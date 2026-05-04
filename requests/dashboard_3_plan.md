# What Ali Wants: Dashboard 3

## Goal
Dashboard 3 = Dashboard 2, but averaged over M independent runs (different seeds), with confidence intervals instead of single lines.

## Before building Dashboard 3: Code correctness verification
Ali wants to manually walk through and verify each component of the existing code (`model.py`, `configs.py`, `plotting.py`) before running at scale. The verification is interactive — Ali drives, Claude assists with short focused answers.

### What to verify
1. **Gradient step**: Is the GD update correct for logistic loss?
2. **Data generation**: Does it match Assumption 1 from the paper (x ~ N(0, Sigma), Pr(y|x) = sigmoid(y x^T w*))?
3. **Empirical loss**: L_hat(w) = (1/n) sum log(1 + exp(-y_i x_i^T w))
4. **Population loss**: MC approximation of L(w) = E[log(1 + exp(-y x^T w))]
5. **Early stopping condition**: Sandwich from Theorem 3.1: L_hat(w_t) <= L_hat(w*_{0:k}) <= L_hat(w_{t-1})
6. **w*_{0:k} definition**: Paper defines it via reordered eigenvalues (pi(i) ordering by lambda_{pi(i)} (u_{pi(i)}^T w*)^2 decreasing). Current code just takes first k components — correct only when eigenvectors align with w* in the right order.
7. **Max-margin direction (w_tilde)**: SVM dual formulation correctness
8. **Step size**: eta <= 1 / (C0 (1 + tr(Sigma) + lambda_1 ln(1/delta)/n))

## Dashboard 3 design (after verification)
- Same metrics as Dashboard 2 (empirical loss, population loss, angle to w*, angle to w_tilde, norm, norm_diff, etc.)
- For each config (n, k, d, w*): run M trajectories with different seeds
- Plot: mean curve + confidence band (e.g., mean +/- 2*std, or percentile bands)
- Sliders for n, k, d as in Dashboard 2; additional slider or input for M
- PKL structure: either one file per seed (reuse existing format) or a bundle file with all M runs

## Resumability requirement
Same as Dashboard 2: each run can be resumed from its pkl. If we ran T steps and want T+T' more, load the pkl and continue.
