# Codebase Overview

## What the code does

You set up a logistic regression problem (x ~ N(0, Sigma), labels via sigmoid), run gradient descent from w=0, and record the trajectory. Then you visualize how the iterates evolve over time.

## The three core files

### 1. `experiment/model.py` — OverparameterizedLogisticRegression class

One class that does everything for a single run:

- **`__init__`**: Stores (n, d, k, eigenvalues, w_star, eta, seed). Initializes empty history lists.
- **`generate_data`**: Draws X from N(0, diag(eigenvalues)) and y from the logistic model. Uses the seed.
- **`empirical_logistic_loss(w)`**: Computes L_hat(w) = (1/n) sum log(1 + exp(-y_i x_i^T w)).
- **`population_logistic_loss(w)`**: MC estimate of L(w) using fresh samples (fixed seed=999).
- **`logistic_gradient(w)`**: Gradient of L_hat. (Used only outside `run_gd`; inside `run_gd` the gradient is inlined.)
- **`run_gd(T)`**: Runs GD from current step to step T. Resumable — if you already ran to step 5000, calling run_gd(10000) continues from 5001. Logs w and loss at log-spaced + linear-spaced checkpoints. Checks the early stopping sandwich condition at every step. Optionally tracks population loss at checkpoints.
- **`compute_max_margin_direction`**: Solves the SVM dual to get w_tilde (max l2-margin direction).
- **Plotting methods**: `plot_dashboard`, `plot_trajectory`, `print_summary` — for quick single-run visualization (matplotlib).

### 2. `experiment/configs.py` — Config helpers

- **`power_law_config(d, k)`**: Returns eigenvalues = i^{-2} and w* = [1,...,1,0,...,0] (k ones).
- **`theoretical_eta(eigenvalues, n)`**: Computes step size upper bound from Theorem 3.1.
- **`create_model(n, d, k, ...)`**: Convenience wrapper — fills in defaults, returns a model.
- **`run_and_save(model)`**: Bundles the model's state into a dict and saves to pkl. The dict contains: config, w_init, w_history, loss_history, pop_loss_history, w_tilde, stopping_times, timestamp.

### 3. `experiment/plotting.py` — Post-hoc metrics and plotting

- **`load_run(filepath)`**: Loads a pkl file.
- **`compute_metrics(data)`**: Takes raw saved data, computes derived metrics: cos(w_t, w*), cos(w_t, w_tilde), angles, norms, and passes through loss/pop_loss/stopping_times.
- **`plot_from_data(data)`**: Matplotlib plots from loaded data.

## Data flow

```
create_model(n, d, k)       → model object
model.generate_data()        → model.X, model.y populated
model.run_gd(T)              → model.w_history, loss_history, stopping_times filled
model.compute_max_margin_direction()  → model.w_tilde
run_and_save(model)          → gd_trajectories/run_n{}_d{}_k{}_seed{}.pkl
```

## Dashboard generation

`experiment/gen_dashboard_2.py` reads all pkl files from `gd_trajectories/`, runs `compute_metrics` on each, bundles the results into JSON, and writes a self-contained HTML file (`dashboard_2/dashboard.html`) with Plotly.

## What's in a pkl file

A dict with keys: config, w_init, w_history, loss_history, pop_loss_history, w_tilde, stopping_times, timestamp. The config sub-dict has: n, d, k, seed, eigenvalues, w_star, eta, num_iterations, track_population_loss, pop_samples_per_dim.

Note: X and y are NOT saved in the pkl (despite what PROJECT_STATE.md says). To reproduce the data, re-create the model with the same seed.

## Existing runs

14 pkl files in `gd_trajectories/`, covering:
- n=100, d=200, k=10
- n=200, d=500, k in {10, 20, 50, 100, 150, 200, 300, 400, 500}
- Various seeds (0, 1, 2, 4)
