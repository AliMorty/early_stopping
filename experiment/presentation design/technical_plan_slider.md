# Technical Plan — Standalone Slider Dashboard (HTML)

## Overview

Two notebooks + one output file:
- **Notebook 1** (`multi_config_experiment.ipynb`) — generate and save trajectory data as `.pkl` files
- **Notebook 2** (`build_dashboard.ipynb`) — load data, compute metrics, build `dashboard.html`
- **Output** (`dashboard.html`) — fully self-contained, works offline on any device

---

## Notebook 2 — `build_dashboard.ipynb`

### Step 1: Load all `.pkl` files
- Scan `gd_trajectories/` for all `run_*.pkl` files
- Load each one using `plotting.load_run(filepath)`

### Step 2: Compute metrics
For each run, compute from raw trajectory data:
- `norm` — `||w_t||` at each checkpoint
- `train_loss` — training loss at each checkpoint (already stored)
- `pop_loss` — population loss at each checkpoint (if available)
- `angle_w_star` — angle between `w_t` and `w*` at each checkpoint
- `angle_w_tilde` — angle between `w_t` and `w̃` at each checkpoint

Reuse logic from `plotting.py` (`compute_metrics`). (PLEASE USE THE SAME CODE if possisble. so if the function is already the same don't write it twice. I just want to make sure you don't make redundant code.)

### Step 3: Group by k/n ratio
- Compute `ratio = k / n` for each run
- Group runs by ratio value
- Sort ratios in ascending order
- Each ratio entry stores: `t_values`, all metric arrays, `stopping_times`, `w_star_norm`, config metadata

### Step 4: Serialize to JSON
Convert all numpy arrays to plain Python lists (JSON-serializable).
Structure:

```json
{
  "ratios": [0.05, 0.10, 0.20, ...],
  "runs": {
    "0.05": {
      "label": "k/n = 0.05  (k=10, n=200, d=500, seed=4)",
      "t_values": [...],
      "norm": [...],
      "train_loss": [...],
      "pop_loss": [...],
      "angle_w_star": [...],
      "angle_w_tilde": [...],
      "stopping_times": [...],
      "w_star_norm": 3.14
    },
    ...
  }
}
```

### Step 5: Build and write `dashboard.html`
Inject into an HTML template:
- The JSON blob as `const DATA = {...}`
- The full Plotly.js source code (bundled, no CDN)
- The dashboard HTML + JS logic

---

## `dashboard.html` — Structure

### Layout
```
[ k/n ratio slider ]

[ Checkboxes ]
  ☑ Norm
  ☑ Angle to w*
  ☐ Angle to w̃
  ☑ Training Loss
  ☐ Population Loss

[ Left plot: current ratio ] [ Right plot: previous ratio ]
[ Left plot: current ratio ] [ Right plot: previous ratio ]
... (one row per checked metric)
```

### Slider behavior
- Discrete steps — only ratios where data exists
- Label: `k/n = 0.05  (k=10, n=200, d=500, seed=4)`
- On slide: right column ← previous left, left column ← new ratio

### Checkbox behavior
- Each checked metric adds a row of two plots (current | previous)
- Unchecking hides the row immediately
- Default: all checked

### Each plot
- X axis: iteration `t`
- Y axis: metric value
- Horizontal dashed line: `||w*||` (norm plots only)
- Vertical red dashed line: theoretical early stopping time `t*`
- Title: metric name + ratio label

---

## Files Involved

| File | Role |
|------|------|
| `gd_trajectories/*.pkl` | Raw trajectory data (input) |
| `plotting.py` | Metric computation (reused) |
| `build_dashboard.ipynb` | Builds the HTML (to be created) |
| `dashboard.html` | Final output (to be created) |

---

## Plotly.js Bundling

- Download `plotly.min.js` once during notebook setup
- Embed full source as `<script>...</script>` block inside the HTML
- Result: single `.html` file, no internet required, works on any device
