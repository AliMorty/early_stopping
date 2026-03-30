# Changes: Store run settings in .pkl files

## What was changed

### 1. `model.py` — inside `run_gd()`

Added two lines right at the start of `run_gd`, before any computation:

```python
self.track_population_loss = track_population_loss
self.pop_samples_per_dim = pop_samples_per_dim
```

**Why:** `run_gd` already received these as arguments, but never stored them on
the model object. By saving them as `self.track_population_loss` and
`self.pop_samples_per_dim`, they become accessible later when `run_and_save`
reads from the model.

---

### 2. `configs.py` — inside `run_and_save()`

Added two entries to the `"config"` dict that gets written into the `.pkl` file:

```python
"track_population_loss": getattr(model, "track_population_loss", None),
"pop_samples_per_dim": getattr(model, "pop_samples_per_dim", None),
```

**Why:** This is where the `.pkl` file is assembled. Adding these keys means
every saved run now records whether population loss was tracked and how many
MC samples per dimension were used.

`getattr(..., None)` is used as a safety fallback — if somehow `run_gd` was
never called before saving, it returns `None` instead of crashing.

---

## What a saved config dict now looks like

```python
{
    "n": 200,
    "d": 500,
    "k": 10,
    "seed": 4,
    "eigenvalues": ...,
    "w_star": ...,
    "eta": 0.18755,
    "num_iterations": 50000,
    "track_population_loss": True,   # <-- new
    "pop_samples_per_dim": 15,       # <-- new
}
```
