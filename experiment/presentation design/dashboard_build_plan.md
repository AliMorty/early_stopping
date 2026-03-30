# Dashboard Build Plan

## What We're Building

A single self-contained `dashboard.html` file. No server, no Python — just open it in a browser. A Python notebook (`build_dashboard.ipynb`) generates it from the `.pkl` files.

---

## Dashboard UI

```
┌──────────────────────────────────────────────────────┐
│  k/n ratio slider: ●───────────────────○             │
│  k/n = 0.10  (k=20, n=200, d=500, seed=2)           │
│                                                      │
│  Metric toggles:                                     │
│  ☑ Norm  ☑ Train Loss  ☐ Pop Loss  ☑ Angle w*  ☐ Angle w̃ │
│                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   Current ratio      │  │   Previous ratio     │   │
│  │                      │  │                      │   │
│  │  All toggled metrics │  │  All toggled metrics │   │
│  │  overlaid on ONE     │  │  overlaid on ONE     │   │
│  │  plot, normalized    │  │  plot, normalized    │   │
│  │                      │  │                      │   │
│  └─────────────────────┘  └─────────────────────┘   │
│                                                      │
│  Vertical dashed red line = early stopping time t*   │
│  Horizontal dashed gray line = ||w*|| (norm ref)     │
└──────────────────────────────────────────────────────┘
```

**Key design choice:** One plot per ratio (not one per metric). All selected metrics share the same plot as overlaid traces. Since metrics have different scales (norm ~3, loss ~0.7, angle ~90°), each metric is **min-max normalized to [0, 1]** so they're visually comparable on the same axes.

---

## Normalization

For each metric array `v`:
```
v_normalized = (v - min(v)) / (max(v) - min(v))
```
This is computed per-metric, per-run. The y-axis label becomes "Normalized value" (unitless). Hovering over a point in Plotly will show the **original** (unnormalized) value in a tooltip — so no information is lost.

---

## Slider Behavior

- Discrete steps — only `k/n` ratios that exist in the data
- Label shows: `k/n = 0.10  (k=20, n=200, d=500, seed=2)`
- On slide:
  - **Left plot** updates to the new ratio
  - **Right plot** updates to the previous slider position (what was just on the left)
- At page load: left = first ratio, right = empty/blank

---

## Checkbox Behavior

- One checkbox per metric: Norm, Train Loss, Pop Loss, Angle to w*, Angle to w̃
- Toggling a checkbox adds/removes that trace from **both** plots immediately
- Default: all available metrics checked
- If a metric doesn't exist for a run (e.g., no population loss), its checkbox is grayed out or the trace simply doesn't appear

---

## Build Pipeline (notebook: `build_dashboard.ipynb`)

### Step 1: Load `.pkl` files
- Scan `gd_trajectories/` for `run_*.pkl`
- Use `plotting.load_run()` — no duplicated code

### Step 2: Compute metrics
- Use `plotting.compute_metrics()` for each run — no duplicated code
- Available metrics from `compute_metrics`: `norms`, `loss_values`, `pop_loss_values`, `angle_w_star`, `angle_w_tilde`

### Step 3: Group by `k/n` ratio
- Compute `ratio = k / n` for each run
- Group runs by ratio (sorted ascending)
- For now: if multiple seeds exist for the same ratio, pick one (or let the user choose later). We won't average.

### Step 4: Serialize to JSON
Convert everything to plain Python lists. Structure:

```json
{
  "ratios": [0.05, 0.10, 0.25, ...],
  "runs": {
    "0.05": {
      "label": "k/n = 0.05  (k=10, n=200, d=500, seed=2)",
      "times": [1, 2, 5, 10, ...],
      "norm": [0.01, 0.05, ...],
      "train_loss": [0.69, 0.68, ...],
      "pop_loss": [0.70, 0.69, ...],
      "angle_w_star": [89.5, 88.1, ...],
      "angle_w_tilde": [45.2, 44.8, ...],
      "stopping_times": [150, 300],
      "w_star_norm": 3.16
    },
    ...
  }
}
```

Normalization happens in JavaScript (client-side), not in the notebook. This way the raw values are available for tooltips.

### Step 5: Build HTML
- Embed JSON as `const DATA = {...}` in a `<script>` tag
- Load Plotly.js from CDN (`<script src="https://cdn.plot.ly/plotly-latest.min.js">`)
  - If offline needed later: bundle it, but CDN is simpler to start
- Write slider + checkbox + plot logic in plain JS
- Save as `experiment/dashboard.html`

---

## Plotly.js Plot Details

Each plot (left and right) is a single Plotly chart with multiple traces:

| Trace | Color | Raw Y | Tooltip |
|-------|-------|-------|---------|
| Norm | green | `norms` | `‖w‖ = 2.45` |
| Train Loss | red | `loss_values` | `Train loss = 0.43` |
| Pop Loss | purple | `pop_loss_values` | `Pop loss = 0.51` |
| Angle to w* | blue | `angle_w_star` | `Angle(w*, w) = 12.3°` |
| Angle to w̃ | orange | `angle_w_tilde` | `Angle(w̃, w) = 8.7°` |

- X-axis: GD iteration `t` (log scale)
- Y-axis: normalized value [0, 1]
- Vertical dashed red line at each stopping time
- Horizontal dashed gray line at normalized position of `‖w*‖` (only meaningful for norm, but shown for reference)

---

## File Summary

| File | Action |
|------|--------|
| `experiment/plotting.py` | No changes — reuse `load_run` and `compute_metrics` |
| `experiment/build_dashboard.ipynb` | Create — loads data, builds HTML |
| `experiment/dashboard.html` | Generated output |

---

## Resolved Decisions

1. **Multiple seeds per ratio** — Add a **dropdown** to select the seed. All seeds for the current ratio are available; switching seed updates both plots.
2. **CDN vs bundled Plotly** — Use **CDN** for now (small file). Offline bundled version can be made later.
3. **Log scale x-axis** — **Yes**, keep log scale for iteration `t` (consistent with existing matplotlib plots).
