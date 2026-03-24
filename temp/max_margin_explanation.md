# `compute_max_margin_direction` — Explanation

The goal is to find the **max-margin direction** $\tilde{w}$ — the unit vector that GD converges to as $t \to \infty$ when the data is linearly separable.

---

## The Primal Problem

The hard-margin SVM finds the weight vector with the largest margin:

$$\min_w \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i x_i^\top w \geq 1 \quad \forall i$$

The direction of the solution is $\tilde{w}$. But solving this directly in $d = 2000$ dimensions is expensive.

---

## The Dual Problem (what the code solves)

The Lagrangian dual of the SVM is:

$$\max_{\alpha \geq 0} \; \sum_i \alpha_i - \frac{1}{2} \alpha^\top (YGY) \alpha$$

where:
- $G = XX^\top \in \mathbb{R}^{n \times n}$ is the **Gram matrix** (only $1000 \times 1000$, much smaller than $d \times d$)
- $(YGY)_{ij} = y_i y_j \, x_i^\top x_j$

The code minimizes the **negative** of this (since `scipy` minimizes):

```python
def dual_objective(alpha):
    return 0.5 * alpha @ YGY @ alpha - np.sum(alpha)
```

Subject to $\alpha_i \geq 0$ (enforced via `bounds = [(0, None)] * n`).

---

## Recovering $w$ from the Dual Solution

Once $\alpha^*$ is found, the primal solution is recovered via the KKT conditions:

$$w_\text{SVM} = X^\top (\alpha^* \odot y) = \sum_i \alpha_i^* y_i x_i$$

```python
w_svm = self.X.T @ (alpha_star * self.y)
```

This is a weighted sum of data points. Only **support vectors** (where $\alpha_i^* > 0$) contribute — all other $\alpha_i^*$ are zero.

---

## Normalization

Finally, normalize to get a unit vector:

$$\tilde{w} = \frac{w_\text{SVM}}{\|w_\text{SVM}\|}$$

```python
self.w_tilde = w_svm / norm(w_svm)
```

---

## Why the Dual?

| | Primal | Dual |
|---|---|---|
| Variables | $w \in \mathbb{R}^d$ | $\alpha \in \mathbb{R}^n$ |
| Size here | $d = 2000$ | $n = 1000$ |

When $n < d$, the dual is cheaper. More generally, the dual only requires inner products $x_i^\top x_j$, which is why it generalizes naturally to kernel SVMs.
