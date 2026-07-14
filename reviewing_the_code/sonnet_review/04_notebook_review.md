# Code Review: multi_config_experiment.ipynb

**File:** `experiment/multi_config_experiment.ipynb`

---

## 1. Cell Structure

The notebook is structured as three independent cells:

| Cell | Purpose | Can run alone? |
|------|---------|----------------|
| Cell 1 (Config) | Define experiment configs and global parameters | Yes |
| Cell 2 (Compute) | Run GD, compute w_tilde, save .pkl files | Requires Cell 1 |
| Cell 3 (Plot) | Load saved .pkl files and render plots | Yes (if .pkl files exist) |

This separation is intentional and correct. The plot cell is self-contained and does not require the models to be in memory.

---

## 2. Cell 1: Config

```python
configs = [
    {"n": 200,  "d": 500,  "k": 10,  "seed": 4},
    {"n": 200,  "d": 500,  "k": 10, "seed": 2},
]
NUM_ITERATIONS = 100000
TRACK_POPULATION_LOSS = True
POP_SAMPLES_PER_DIM = 25
```

The parameters are passed correctly to `create_model(**cfg)` in Cell 2. `n`, `d`, `k`, and `seed` are the four expected keys, and additional keys (`eigenvalues`, `w_star`, `eta`) are optional and default to `power_law_config` + `theoretical_eta` if not provided. **Correct.**

The two active configs use the same `(n, d, k)` but different seeds. This is a valid setup for checking seed-to-seed robustness.

---

## 3. Cell 2: Compute

```python
model = create_model(**cfg)
model.generate_data()
model.run_gd(NUM_ITERATIONS,
             track_population_loss=TRACK_POPULATION_LOSS,
             pop_samples_per_dim=POP_SAMPLES_PER_DIM)
model.compute_max_margin_direction()
run_and_save(model, save_dir="gd_trajectories")
```

**Order of operations is correct:** data must be generated before GD runs, and `w_tilde` must be computed before saving (since it is included in the pkl).

**One issue:** `run_gd` is called before `compute_max_margin_direction`. This means that during the GD run, `model.w_tilde` is `None`, so no angle-to-w_tilde is computed inside the GD loop. However, `w_tilde` is only needed at plot time (computed from `w_history`), not during the GD run itself. The `w_history` stores the full trajectory and `plotting.py` computes the angle post-hoc. **Not a bug.**

---

## 4. Cell 3: Plot

```python
import importlib, plotting
importlib.reload(plotting)
```

The `importlib.reload` ensures that any updates to `plotting.py` made after the kernel was started are picked up without requiring a full kernel restart. **Good practice.**

```python
pkl_files = sorted(glob.glob("gd_trajectories/run_*.pkl"))
for filepath in pkl_files:
    data = load_run(filepath)
    plot_from_data(data)
```

The glob pattern `run_*.pkl` picks up all saved runs in `gd_trajectories/`. Files are sorted alphabetically before plotting. **Correct.**

**Note:** The working directory matters here. The cell uses a relative path `"gd_trajectories/run_*.pkl"`. This will only work if the notebook is run from within `experiment/`. If the kernel is started from the repo root, the path will not resolve. Jupyter typically sets the CWD to the notebook's directory, so this is usually fine -- but worth knowing.

---

## 5. Overall Flow Gaps

- **No stopping time computed before saving:** If the run finishes without any stopping time being detected (e.g., $T$ is too small or the loss never crosses below $\widehat{\mathcal{L}}(w^*_{0:k})$), `stopping_times` will be an empty list in the pkl, and no red line will appear in plots. This is not a bug but can be surprising. It means $T$ should be set large enough for the loss to descend past the truncated reference.

- **No resume support yet:** The notebook has no `resume_from_pkl` flow (discussed in session notes as future work). Running Cell 2 again on the same config will overwrite the existing pkl. See also the filename issue noted in `02_configs_review.md`.

---

## 6. Suggested Tests

- **Verify stopping time is in range:** After running a config, print `model.stopping_times`. If it is empty, increase `NUM_ITERATIONS`. If it has many entries (more than ~5), this might indicate the loss is oscillating around $\widehat{\mathcal{L}}(w^*_{0:k})$, which would be unexpected for a smooth monotonically decreasing GD run.

- **Overparameterization check:** Before running, verify that `d >= n` for all configs. The paper's guarantees only apply in the overparameterized regime. The active configs have $n=200$, $d=500$, which gives $d/n = 2.5$. **This is fine.**

- **Population loss U-shape:** With `TRACK_POPULATION_LOSS=True`, inspect the population loss curve. It should decrease initially (model is learning signal), reach a minimum near the theoretical stopping time, then increase (model is drifting toward $\tilde{w}$). This is the central empirical prediction from the paper.

- **Reproducibility check:** Run Cell 2 twice with the same config. The second run will overwrite the first pkl. Re-run Cell 3 and confirm the plots are identical (since the same seed produces the same data).
