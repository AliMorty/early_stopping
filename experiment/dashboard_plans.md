# Dashboard Plans

## dashboard_2 — Single-Run Explorer
- **Folder:** `dashboard_2/`
- **Generator:** `experiment/gen_dashboard_2.py`
- **Description:** Interactive explorer for individual experiment runs.
- **Sliders:**
  - **seed** — slide across different random seeds
  - **k/n ratio** — slide across different k/n values
- **Display:** For each (seed, k/n) pair, shows all metric curves for that single run (norm, norm_diff, losses, angles) with per-metric toggles for visibility, normalization, and argmin.

## dashboard_3 — Multi-Run Explorer
- **Folder:** `dashboard_3/`
- **Description:** Aggregated view across many seeds for a fixed configuration (n, d, Σ, w*).
- **Slider:**
  - **k/n ratio** — slide across different k/n values
- **Display modes:**
  1. **Averaged view** — mean curves with confidence intervals (across seeds) for each metric
  2. **Seed slider** — slide through individual seed runs for the selected k/n
