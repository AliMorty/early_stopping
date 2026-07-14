# Large-Scale Experiments Plan

## Goal
Run many seeds for fixed configurations (n, k, d, Σ, w*) overnight, generate `.pkl` files, then build dashboards that aggregate across seeds.

---

## Phase 1: Code Review (before running anything)

### 1.1 Core correctness (`model.py`)
- **Data generation**: Verify `x ~ N(0, Σ)` sampling is correct (diagonal covariance shortcut: `Z * sqrt(λ)`).
- **Label generation**: Check `Pr(y|x) = sigmoid(x^T w*)` with y ∈ {-1, +1}.
- **Gradient computation**: Verify gradient of logistic loss matches the formula. Check sign conventions (−1/+1 labels vs 0/1).
- **Loss computation**: Confirm empirical loss `(1/n) Σ log(1 + exp(-y_i x_i^T w))` is correct.
- **Population loss**: The MC approximation uses a fixed seed (999) — is this a problem when averaging over training seeds? Should it use a separate fresh seed per evaluation?
- **Stopping time condition**: Currently checks `L(w_{t-1}) ≤ L(w*_{0:k}) ≤ L(w_{t-2})`. Verify this matches the paper's definition exactly (off-by-one is easy to get wrong here).
- **Checkpoint logging**: The loss recorded at checkpoint `t` is actually computed *before* the gradient step at `t` (i.e., it's `L(w_{t-1})`). Verify this is intentional and consistent with what the dashboard displays.
- **Max-margin solver**: Check that the dual SVM formulation and the extraction of `w_tilde` are correct.

### 1.2 Configs and saving (`configs.py`)
- **`power_law_config`**: Verify eigenvalue decay `λ_i = i^{-2}` and w* construction.
- **`theoretical_eta`**: Check step size formula matches Theorem 3.1 from the paper.
- **`run_and_save`**: Confirm all necessary data is saved (w_history, loss_history, pop_loss_history, stopping_times, config). Check nothing is accidentally shared/mutated between saves.

### 1.3 Numerical stability
- **Overflow in `exp(margins)`**: For large `||w||`, margins can be huge. Check `logaddexp` usage protects against this.
- **Gradient sigmoid**: `−1/(1+exp(margins))` — for large negative margins, `exp(margins) → 0`, so this → −1. For large positive margins, `exp(margins) → ∞`, need to check no overflow.
- **Norm of `w_tilde`**: If SVM solver returns near-zero solution, `w/||w||` would blow up. Is there a guard?

### 1.4 Reproducibility
- **Seed handling**: `np.random.RandomState(seed)` is used for data generation — good, isolated from global state. But does anything else use the global RNG?
- **Determinism**: Confirm that running the same (n, d, k, seed) twice produces identical `.pkl` output.

---

## Phase 2: Batch Runner Script

### 2.1 Design
- A Python script (e.g., `run_batch.py`) that takes a config (n, d, k, eigenvalue scheme, w* scheme) and a list/range of seeds.
- Loops over seeds, runs `create_model` → `generate_data` → `run_gd` → `compute_max_margin_direction` → `run_and_save` for each.
- Saves all `.pkl` files into a structured directory, e.g.:
  ```
  gd_trajectories/batch_<timestamp>/
      run_n200_d500_k5_seed0.pkl
      run_n200_d500_k5_seed1.pkl
      ...
  ```
- Logs progress to a file so we can check status in the morning.
- Handles crashes gracefully: if seed `i` fails, log the error and continue to seed `i+1`.

### 2.2 Parameter grid
- Fix: n, d, eigenvalue scheme, w* scheme
- Vary: k (multiple k/n ratios), seed (e.g., 0–49 for 50 seeds)
- This gives us one batch per (n, d, Σ, w*) setting, with k and seed as the two axes.

---

## Phase 3: Dashboard Generation

### 3.1 dashboard_2 — Single-Run Explorer (see `experiment/dashboard_plans.md`)
- Add sliders for seed and k/n ratio to the existing single-run dashboard.
- Loads all `.pkl` files from a batch directory.
- User slides to pick a (seed, k/n) pair and sees the full metric dashboard for that run.

### 3.2 dashboard_3 — Multi-Run Explorer (see `experiment/dashboard_plans.md`)
- Slider for k/n ratio.
- For each k/n, aggregates across all seeds:
  - **Averaged view**: mean ± confidence interval for each metric curve.
  - **Seed slider**: browse individual seed runs within the selected k/n.

---

## Phase 4: Overnight Execution

### Pre-flight checklist
1. Code review complete (Phase 1) — no known bugs in core logic.
2. Batch runner tested on 2–3 seeds with small T to verify output correctness.
3. Sanity check: load a test `.pkl`, verify values look reasonable.
4. Disk space check: estimate `.pkl` size × number of runs.
5. Logging enabled so progress can be checked remotely.

### Run
- Start the batch runner via `nohup` or `tmux` so it survives terminal disconnection.
- Redirect stdout/stderr to a log file.

### Morning check
- Verify all expected `.pkl` files exist.
- Spot-check a few for correctness (reasonable loss values, stopping times found, etc.).
- Generate dashboards.
