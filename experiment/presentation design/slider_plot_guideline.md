# Slider Plot Design — GD Trajectory Dashboard

## Goal

A standalone `dashboard.html` file that visualizes GD trajectories across
different k/n ratios interactively. No Python or Jupyter needed to view it.
Openable on any browser, including iPad.

---

## Data Structure

Trajectories are saved as `.pkl` files in `gd_trajectories/`.
**Current scope:** `n` and `d` are fixed. Only `k` varies across runs.
The slider represents `k/n` ratio — not raw `k`.

For the dashboard, runs are grouped by their `k/n` ratio.
If multiple seeds exist for the same ratio, they are averaged or shown as
separate traces (TBD).

**Future extension (not now):** allow sliding over `n`, `d`, or other parameters.

---

## Sliders

### Primary slider: k/n ratio
- Discrete steps (only ratios where data exists)
- Moving the slider updates all plots to show the trajectory for that ratio
- Label format: `k/n = 0.05  (k=10, n=200)`

### Secondary controls (future / optional)
- Seed selector (dropdown or radio buttons) — if multiple seeds per ratio exist
- Metric selector (dropdown): norm, training loss, population loss, angle to w*, angle to w̃

---

## Plot Layout

Two plots side by side:

| Left | Right |
|------|-------|
| Current ratio | Previous ratio |

When the slider moves:
- Right plot ← what was on the left
- Left plot ← new ratio

This allows direct visual comparison between consecutive ratios.

### Each plot shows (initially: norm trajectory)
- X axis: GD iteration `t`
- Y axis: metric value (e.g. `||w_t||`)
- Horizontal dashed line: `||w*||` (target norm)
- Vertical red dashed line: theoretical early stopping time `t*`
- Title: `k/n = {ratio}  (k={k}, n={n}, d={d}, seed={seed})`

---

## Technical Implementation

### Notebook 2 — Build HTML
1. Load all `.pkl` files from `gd_trajectories/`
2. Compute metrics for each run (reuse `plotting.py` logic)
3. Group runs by `k/n` ratio
4. Serialize all data to a JSON blob
5. Inject JSON + Plotly.js into an HTML template
6. Write `dashboard.html`

### HTML / JS
- **Plotly.js** for plots (CDN or bundled)
- Plain JS for slider logic
- All data embedded as a `const DATA = {...}` block in the HTML
- No server, no backend, fully self-contained

---

## Open Questions

- [ ] If multiple seeds exist for the same ratio, show all traces on one plot or average them?
- [ ] Which metric is shown by default when the page loads?
- [ ] Should the stopping time line always be shown, or toggleable?
- [ ] Should Plotly.js be bundled into the HTML (larger file, fully offline) or loaded from CDN (requires internet)?
