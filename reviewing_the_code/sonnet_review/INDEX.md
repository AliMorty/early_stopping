# Code Review Index

Reviews of the core experiment pipeline for the early stopping project.
Each file covers one component.

---

| File | Component | Key Findings |
|------|-----------|--------------|
| [01_model_review.md](01_model_review.md) | `model.py` | Data generation and gradient are correct. Early stopping condition matches Theorem 3.1 exactly. Minor: `loss_history` stores loss of the previous iterate, not the current one (off by one step). Custom `w*` configs may miscompute `w_star_k`. |
| [02_configs_review.md](02_configs_review.md) | `configs.py` | Step size formula matches Theorem 3.1. `C0=2.0` is a valid but arbitrary choice. `run_and_save` does not store `X` and `y`. Same-config runs silently overwrite each other due to the filename format. |
| [03_plotting_review.md](03_plotting_review.md) | `plotting.py` | The previously reported bug (missing angle fields) is fixed. Angle computation is correct. Minor: the first stopping time line is drawn twice, making it slightly darker. |
| [04_notebook_review.md](04_notebook_review.md) | `multi_config_experiment.ipynb` | Cell structure and order of operations are correct. Relative path in Cell 3 requires the notebook to run from `experiment/`. No resume support yet. |

---

## Summary of Issues Found

### Real bugs
- None found that affect results.

### Minor inconsistencies
1. **`loss_history` off by one** (`model.py`): The loss stored at step $t$ is $\widehat{\mathcal{L}}(w_{t-1})$, not $\widehat{\mathcal{L}}(w_t)$. Does not affect stopping time detection. Only matters if you cross-reference `w_history` and `loss_history` directly.
2. **Double stopping time line** (`plotting.py`): `stopping_times[0]` is drawn twice, making the first red dashed line slightly darker than the rest. Cosmetic only.

### Design gaps (not bugs)
3. **`X` and `y` not saved** (`configs.py`): The pkl file does not include training data. Reproducibility relies on seed determinism.
4. **Filename collision** (`configs.py`): Re-running the same config silently overwrites the existing pkl. `T` and timestamp are absent from the filename.
5. **`w_star_k` assumes natural ordering** (`model.py`): For custom `w*` where informative components are not in positions $1, \ldots, k$, the truncated reference vector will be incorrect. Safe for `power_law_config`.

---

## Suggested Priority Tests

1. Numerical gradient check on `logistic_gradient` using `scipy.optimize.check_grad`.
2. Verify the stopping time sandwich condition manually on a short run.
3. Confirm the population loss shows a U-shaped curve (decreasing then increasing) with the minimum near the theoretical stopping time.
4. Run the same config twice and confirm the overwrite behavior.
