# My Understanding of the Project

## The Paper Being Studied

The reference paper is **"Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression"** (Wu, Bartlett, Telgarsky, Yu — ICML 2025).

The setting: binary classification with logistic regression in the **overparameterized regime** (more parameters than training samples, d ≥ n). Features are Gaussian with an anisotropic covariance Σ, and labels come from a true logistic model parameterized by w*.

The core phenomenon: when you run gradient descent (GD) on the training loss, the iterates **diverge in norm** but **converge in direction** toward the **maximum ℓ₂-margin solution** (called w̃). This is the "implicit bias" of GD. The paper's central claim is that **this asymptotic behavior is bad** — and that stopping early is statistically better.

---

## The Three Main Theoretical Results

1. **Early stopping gives calibration; running to convergence does not.**
   Early-stopped GD achieves vanishing excess logistic risk and its predicted probabilities converge to the true conditional probabilities. Asymptotic GD (w̃ direction) has *unbounded* logistic risk and a constant calibration error — it is poorly calibrated no matter how much data you have.

2. **Early stopping needs polynomially many samples; interpolation needs exponentially many.**
   To achieve small zero-one error, early-stopped GD needs only O(poly(n)) samples. Any interpolating estimator — including the asymptotic GD solution — requires exponentially many samples. This is a sharp separation.

3. **Early stopping is connected to ℓ₂-regularization.**
   The GD path and the ℓ₂-regularization path stay close: angle ≤ π/4, and their norms differ by at most a factor of ~3.4. This partially explains *why* early stopping helps — it implicitly mimics the effect of explicit regularization.

---

## The Experimental Setup

The simulation models the paper's setting:
- **d = 2000** (dimensions), **n = 1000** (samples) — overparameterized by factor 2
- **k = 100** — the number of "informative" dimensions; w* has 1.0 in its first k components, 0 elsewhere
- **Σ** — diagonal with eigenvalues decaying as i⁻², so most variance is in early dimensions
- GD starts from w₀ = 0, uses a step size derived from Theorem 3.1

Three quantities are tracked over GD iterations:
- **angle(wₜ, w*)** — how aligned is the current iterate with the true parameter
- **‖wₜ‖ − ‖w*‖** — how far the norm of the iterate is from the true parameter's norm
- **angle(wₜ, w̃)** — how aligned the iterate is with the max-margin direction

The **theoretical stopping time** from Theorem 3.1 is the first iteration t where:
> L̂(wₜ) ≤ L̂(w*₀:ₖ) ≤ L̂(wₜ₋₁)

i.e., the empirical loss of the iterate crosses below the empirical loss of the k-truncated true parameter. This is marked as a vertical red dashed line on all plots.

---

## The Central Empirical Question

> Do the argmins of the three tracked quantities (angle to w*, norm diff, angle to w̃) coincide at approximately the same iteration — and does that iteration match the theoretical stopping time?

If yes, it suggests there is a single natural "sweet spot" for early stopping that multiple signals agree on, and that the theoretical criterion from the paper identifies it correctly.

---

## The Bigger Research Goals

The project is pursuing three questions, in order of ambition:

1. **Population GD vs empirical GD** — does the early stopping phenomenon still appear if you use the true population gradient instead of the empirical gradient? If population GD does not drift toward w̃, then the drift is purely an artifact of overfitting the finite training set.

2. **Does E[w̃] converge to w*?** — Across many independent datasets, does the average max-margin direction point toward w*? Preliminary results at M=50 are promising but inconclusive.

3. **Data-dependent early stopping (the main goal)** — The paper's stopping time requires knowing w* and Σ, which a practitioner doesn't have. The real question is: *can we design a stopping rule using only observable training quantities that achieves the same statistical benefits?* Possible ideas include validation loss monitoring, tracking the rate of change of the iterate direction, or stability across bootstrap resamples.

---

## Infrastructure

A pipeline has been built to run experiments:
- **`model.py`** — core GD simulation class; stores full trajectory (w_history, loss_history, stopping_times)
- **`configs.py`** — helpers for creating configs, computing theoretical step size, saving/loading .pkl files
- **`plotting.py`** — loads .pkl files, computes metrics (angles, norms) at plot time, generates plots
- **`multi_config_experiment.ipynb`** — define configs → run GD → save .pkl files → plot
- **Dashboard (in progress)** — an interactive HTML dashboard with Plotly.js, sliders for k/n ratio, checkboxes for metrics, side-by-side current vs previous comparison

---

## Current State (as of 2026-03-30)

The experiment pipeline is functional. The dashboard design is documented but not yet implemented. The main next step is building `build_dashboard.ipynb` and `dashboard.html`.

The overarching long-term goal — a practical, data-dependent early stopping rule — has not yet been explored experimentally.

---

*Written by Claude Sonnet 4.6 based on reading `references/main_paper.md` and all session summaries.*
