# Code Review: configs.py

**File:** `experiment/configs.py`

---

## 1. `power_law_config(d, k)`

Generates the default eigenvalue structure and true parameter vector.

```python
eigenvalues = np.array([(i + 1) ** (-2) for i in range(d)])
w_star = np.zeros(d)
w_star[:k] = 1.0
```

This sets $\lambda_i = i^{-2}$ for $i = 1, \ldots, d$, consistent with the source and capacity conditions in Eq. (1) of the paper (with $a = 2$):

$$\lambda_i \asymp i^{-a}, \quad a > 1$$

The true parameter has $w^*_i = 1$ for $i \leq k$ and $0$ elsewhere.

For this configuration, the relevance ordering $\pi(i)$ defined in the paper (sorting by $\lambda_{\pi(i)} (u_{\pi(i)}^\top w^*)^2$) coincides with the natural ordering $1, 2, \ldots, k$, since components $k+1, \ldots, d$ contribute zero signal. **Correct for this specific config.**

**Note:** If a custom $w^*$ is used where signal is not in the first $k$ components, the truncation logic in `model.py` (`w_star_k[:k] = w_star[:k]`) will not produce the correct $w^*_{0:k}$. See the model review for details.

---

## 2. `theoretical_eta(eigenvalues, n, C0=2.0, delta=0.01)`

Theorem 3.1 requires the step size to satisfy:

$$\eta \leq \frac{1}{C_0 \left(1 + \mathrm{tr}(\boldsymbol{\Sigma}) + \lambda_1 \ln(1/\delta) / n\right)}$$

where $C_0 > 1$ is a universal constant.

**Implementation:**
```python
tr_sigma = np.sum(eigenvalues)
lambda_1 = eigenvalues[0]
return 1.0 / (C0 * (1 + tr_sigma + lambda_1 * np.log(1.0 / delta) / n))
```

This matches the theorem formula exactly. **Correct.**

**On the choice of C0 = 2.0:** The paper states $C_0 > 1$ is a universal constant but does not pin down its value. The code uses $C_0 = 2$, which satisfies the $C_0 > 1$ requirement. This is a reasonable default but is not derived from first principles. The step size will be valid (conservative), but a smaller $C_0$ closer to 1 would give a larger (less conservative) step size. This is worth being aware of when comparing step sizes across different setups.

---

## 3. `create_model(n, d, k, ...)`

A convenience wrapper that fills in defaults via `power_law_config` and `theoretical_eta`.

```python
if eigenvalues is None and w_star is None:
    eigenvalues, w_star = power_law_config(d, k)
elif eigenvalues is None or w_star is None:
    raise ValueError("Provide both eigenvalues and w_star, or neither.")
```

The guard correctly catches the case where only one of `eigenvalues` / `w_star` is provided. **Correct.**

---

## 4. `run_and_save(model, save_dir)`

Saves the model state to a `.pkl` file after GD has finished.

**What IS saved:**
- Full config (n, d, k, seed, eigenvalues, w_star, eta, num_iterations, track_population_loss, pop_samples_per_dim)
- `w_init` (always zeros, hardcoded as `np.zeros(model.d)`)
- `w_history` (list of (t, w_t) tuples)
- `loss_history` (list of (t, loss) tuples)
- `pop_loss_history` (if available)
- `w_tilde` (max-margin direction, if computed)
- `stopping_times` (list of detected stopping iterations)
- `timestamp`

**What is NOT saved:** The training data `X` and `y`.

This means if you need to resume a run (continue GD from where it left off), the model cannot re-use the same dataset -- it would re-generate data from the seed, which gives the same `X` and `y` provided the seed is deterministic. So in practice this is fine as long as `generate_data()` is always called with the same seed before resuming. However, if the plan is ever to support `resume_from_pkl` without re-generating data, `X` and `y` would need to be stored. This was flagged as a planned feature in the session notes (2026-03-24 session 5) but has not been implemented yet.

**Filename format:**
```python
filename = f"run_n{model.n}_d{model.d}_k{model.k}_seed{model.seed}.pkl"
```

The planned format discussed in the session notes was `run_n{n}_d{d}_k{k}_seed{seed}_T{T}_{timestamp}.pkl`. The current format omits `T` and `timestamp` from the filename. As a result, running the same config twice will **overwrite** the previous file. The timestamp is stored inside the pkl but is not visible from the filename. If you want to keep multiple runs (e.g., different T values), this will cause silent data loss.

---

## 5. Suggested Tests

- **Step size sanity check:** For a standard config (e.g., $n=1000$, $d=2000$, power law), print `theoretical_eta(eigenvalues, n)` and verify it is small but positive. Since $\mathrm{tr}(\Sigma) = \sum_{i=1}^{d} i^{-2} \approx \pi^2/6 \approx 1.645$ and $\lambda_1 = 1$, the denominator should be approximately $C_0 \cdot (1 + 1.645 + 1 \cdot \ln(100)/1000) \approx C_0 \cdot 2.65$, giving $\eta \approx 0.19$ for $C_0 = 2$.

- **Overwrite risk:** Run `run_and_save` twice with the same config. Confirm the second call silently overwrites the first file. Decide if this is acceptable behavior or if a uniqueness check should be added.

- **Resume without X/y:** Verify that loading a pkl and calling `generate_data()` with the stored seed produces identical `X` and `y`. This confirms that data reproducibility via seed is sufficient as a substitute for saving X and y.
