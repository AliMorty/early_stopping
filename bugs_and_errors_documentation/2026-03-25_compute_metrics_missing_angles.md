# Bug Report — `compute_metrics` missing angle fields

**Date observed:** 2026-03-25
**File affected:** `experiment/plotting.py` — `compute_metrics()`

---

## What Was Wrong

`compute_metrics()` returns `cos_wstar` and `cos_wtilde` (raw cosine similarity
values), but never computes the corresponding angles in degrees.

The angle conversion existed only inside `plot_from_data()`, buried inline:

```python
angle_wstar = np.degrees(np.arccos(np.clip(metrics["cos_wstar"], -1, 1)))
```

This means any code that calls `compute_metrics()` directly (e.g. the dashboard
notebook) cannot get angles without duplicating this conversion logic — which
violates the design principle of keeping all metric computation in one place.

---

## Why It Matters

The HTML dashboard (`build_dashboard.ipynb`) needs to call `compute_metrics()`
and get all plottable quantities ready to serialize to JSON. If angles are
missing, either the dashboard would have to duplicate the conversion or angles
would be unavailable.

---

## Fix Applied

Added `angle_w_star` and `angle_w_tilde` (in degrees) directly to the dict
returned by `compute_metrics()`, computed from the already-present cosine values:

```python
metrics["angle_w_star"] = np.degrees(np.arccos(np.clip(metrics["cos_wstar"], -1, 1)))
if "cos_wtilde" in metrics:
    metrics["angle_w_tilde"] = np.degrees(np.arccos(np.clip(metrics["cos_wtilde"], -1, 1)))
```

`plot_from_data()` was updated to use `metrics["angle_w_star"]` and
`metrics["angle_w_tilde"]` directly instead of recomputing them inline.

---

## Other Things Reviewed (No Bugs Found)

- `w_tilde` is stored as a unit vector (`w_svm / norm(w_svm)` in `model.py`
  line 187), so all dot products against it are correct cosine similarities.
- `times` and `loss_times` come from the same checkpoints — consistent.
- Early stopping two-sided condition logic in `run_gd` is correct.
- `pop_loss_history` saved as `None` when empty — handled correctly in
  `run_and_save`.
