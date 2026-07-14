# Code Review: plotting.py

**File:** `experiment/plotting.py`

---

## 1. `load_run(filepath)`

Simple pickle loader. No issues. The file is opened in binary read mode (`"rb"`), which is correct for pickle.

---

## 2. `compute_metrics(data)`

This is the most important function in the file. It was previously the source of a known bug (missing `angle_w_star` and `angle_w_tilde` fields). The fix has been applied and is reviewed here.

### Angles to w*

The angle between $w_t / \|w_t\|$ and $w^* / \|w^*\|$ in degrees:

$$\theta_{w^*}(t) = \frac{180}{\pi} \cdot \arccos\!\left(\frac{w_t}{\|w_t\|} \cdot \frac{w^*}{\|w^*\|}\right)$$

```python
w_star_dir = w_star / norm(w_star)
...
cos_wstar.append(np.dot(w_dir, w_star_dir))
...
metrics["angle_w_star"] = np.degrees(np.arccos(np.clip(metrics["cos_wstar"], -1, 1)))
```

The `np.clip(..., -1, 1)` guard prevents `arccos` from receiving values slightly outside $[-1, 1]$ due to floating point rounding. **Correct. Bug is fixed.**

### Angles to $\tilde{w}$

$$\theta_{\tilde{w}}(t) = \frac{180}{\pi} \cdot \arccos\!\left(\frac{w_t}{\|w_t\|} \cdot \tilde{w}\right)$$

```python
if w_tilde is not None:
    metrics["cos_wtilde"] = np.array(cos_wtilde)
    metrics["angle_w_tilde"] = np.degrees(np.arccos(np.clip(metrics["cos_wtilde"], -1, 1)))
```

`w_tilde` is already a unit vector (normalized in `compute_max_margin_direction`), so no renormalization is needed here. **Correct. Bug is fixed.**

### Loss histories

Both empirical and population loss are extracted from their respective history lists, filtering out $t = 0$ entries. This is consistent with how `w_history` handles $t = 0$ (also skipped). **Correct.**

### Stopping times

```python
stopping_times = data.get("stopping_times")
if stopping_times is not None and len(stopping_times) > 0:
    metrics["stopping_times"] = stopping_times
```

If no stopping time was detected (empty list), the key is omitted from the metrics dict. This causes downstream code to use `metrics.get("stopping_times")` safely. **Correct.**

---

## 3. `plot_from_data(data)`

### Plot availability logic

The function dynamically determines which plots to render:

```python
if "norms" in metrics: available.append("norm")
if "loss_values" in metrics or "pop_loss_values" in metrics: available.append("loss")
if "cos_wstar" in metrics: available.append("angle")
```

`cos_wstar` is always computed when `w_history` is non-empty, so the angle plot always appears. `angle_w_tilde` appears conditionally inside the angle panel if `w_tilde` was saved. **Correct.**

### Stopping time lines (`add_stopping_lines`)

```python
def add_stopping_lines(ax):
    if stopping_times is not None:
        for st in stopping_times:
            ax.axvline(st, color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(stopping_times[0], color='red', linestyle='--', alpha=0.6,
                   linewidth=1.5, label=f'stopping time (first={stopping_times[0]})')
```

**Minor visual issue:** `stopping_times[0]` receives two `axvline` calls -- one from the loop (alpha=0.4) and one from the explicit call below (alpha=0.6). This causes the first stopping time line to appear slightly darker/thicker than intended. It does not affect correctness. If there is only one stopping time, this just draws the same line twice.

### Grid layout

```python
ncols = min(n_plots, 2)
nrows = (n_plots + 1) // 2
```

For 3 plots (the typical case): `ncols=2`, `nrows=2`, giving a 2x2 grid with one unused panel. The unused panel is hidden via `axes[i].set_visible(False)`. **Correct.**

### `w_star_norm` horizontal line on norm plot

```python
ax.axhline(metrics["w_star_norm"], ...)
```

This draws a horizontal line at $\|w^*\|$. For the default power_law_config with $k$ ones as the first $k$ components: $\|w^*\| = \sqrt{k}$. This lets you visually check whether $\|w_t\|$ reaches $\|w^*\|$ around the theoretical stopping time. **Correct.**

---

## 4. Suggested Tests

- **Angle range check:** After running `compute_metrics`, verify that all values in `metrics["angle_w_star"]` and `metrics["angle_w_tilde"]` are in $[0, 180]$. If any values fall outside this range, the `arccos` input was not properly clipped.

- **Stopping time on plot vs detection:** After a run, confirm visually that the red dashed vertical line on the loss plot falls at the point where the training loss curve crosses below $\widehat{\mathcal{L}}(w^*_{0:k})$. You can draw a horizontal line at the truncated loss value to make this check easier.

- **Double line issue:** If you have only one stopping time, open the plot and zoom into the red dashed line. You will see it is slightly darker than expected because two lines are drawn. This is cosmetic but worth confirming.

- **Population loss U-shape:** When `track_population_loss=True`, verify that `metrics["pop_loss_values"]` shows the expected U-shaped curve: decreasing early (as the model learns signal), then increasing later (as the iterate drifts toward $\tilde{w}$ and overfits). This is the key empirical result from Figure 1 of the paper.
