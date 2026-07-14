# Is the Code's w*_{0:k} Correct?

## What the Paper Defines

The paper's **Additional Notation** (Section 2) defines $w^*_{0:k}$ as follows.

Let $(\lambda_i)_{i \geq 1}$ be the eigenvalues of $\boldsymbol{\Sigma}$, sorted non-increasingly, with corresponding eigenvectors $\mathbf{u}_i$. Define the re-sorted index sequence $(\pi(i))_{i \geq 1}$ such that

$$\lambda_{\pi(i)} \left(\mathbf{u}_{\pi(i)}^\top \mathbf{w}^*\right)^2 \quad \text{is non-increasing in } i.$$

Then:

$$\mathbf{w}^*_{0:k} := \sum_{i \leq k} \mathbf{u}_{\pi(i)} \mathbf{u}_{\pi(i)}^\top \mathbf{w}^*$$

In words: $w^*_{0:k}$ is the projection of $w^*$ onto the $k$ eigenvectors that contribute the most **signal**, where signal is measured by $\lambda_j (u_j^\top w^*)^2$ -- the product of the eigenvalue (how much variance the data has in that direction) and the squared component of $w^*$ in that direction.

This quantity is the per-dimension signal-to-noise ratio. The $\pi$ ordering ranks dimensions from most informative to least informative.

---

## What the Code Does

```python
w_star_k = np.zeros(self.d)
w_star_k[:self.k] = self.w_star[:self.k]
```

This simply takes the first $k$ components of $w^*$ in the natural coordinate ordering.

---

## When Are These the Same?

Since the code works in the eigenbasis of $\Sigma$ (i.e., $\Sigma = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$ and $U = I$), the eigenvectors are the standard basis vectors $u_i = e_i$. So $u_i^\top w^* = w^*_i$, and the signal quantity becomes:

$$\lambda_{\pi(i)} \left(w^*_{\pi(i)}\right)^2$$

The $\pi$ ordering ranks dimensions by $\lambda_j (w^*_j)^2$ from largest to smallest.

For the code's approach to match the paper's definition, we need:

$$\pi(i) = i \quad \text{for all } i \leq k$$

That is, the $k$ most informative dimensions must be exactly dimensions $1, 2, \ldots, k$ in that order. This requires $\lambda_j (w^*_j)^2$ to be non-increasing for $j = 1, \ldots, k$ AND all uninformative dimensions ($j > k$, where $w^*_j = 0$) must rank after all informative ones.

---

## Does the Default Config Satisfy This?

For `power_law_config`:
- $\lambda_j = j^{-2}$, already non-increasing ✓
- $w^*_j = 1$ for $j \leq k$, $w^*_j = 0$ for $j > k$
- Signal in each dimension: $\lambda_j (w^*_j)^2 = j^{-2}$ for $j \leq k$, and $0$ for $j > k$
- The sequence $1, 1/4, 1/9, \ldots, k^{-2}, 0, 0, \ldots$ is already non-increasing ✓

So for `power_law_config`, the natural ordering coincides with the $\pi$ ordering. **The code is correct for this config.**

---

## When Would the Code Be Wrong?

### Case 1: Non-uniform $w^*$ components

Suppose you change $w^*$ so that signal is concentrated in a later dimension. For example:

$$w^*_1 = 0.01, \quad w^*_2 = 10, \quad w^*_j = 0 \text{ for } j > 2, \quad k = 1$$

Signal per dimension:
- Dimension 1: $\lambda_1 (w^*_1)^2 = 1 \cdot 0.0001 = 0.0001$
- Dimension 2: $\lambda_2 (w^*_2)^2 = 0.25 \cdot 100 = 25$

The paper's $w^*_{0:1}$ would include dimension 2 (it has far more signal). The code would include dimension 1. These are completely different vectors, and the stopping condition would fire at the wrong iteration.

### Case 2: Spiked covariance where eigenvalue order disagrees with signal order

Suppose:
- $w^*_1 = 1, w^*_2 = 1, k = 1$
- $\lambda_1 = 100$ (a spike), $\lambda_2 = 0.01$

Signal:
- Dimension 1: $100 \cdot 1 = 100$
- Dimension 2: $0.01 \cdot 1 = 0.01$

Here the code and paper agree -- dimension 1 has more signal.

But now suppose the spike is in a later dimension:
- $\lambda_1 = 0.01$, $\lambda_2 = 100$
- $w^*_1 = 1, w^*_2 = 1, k = 1$

Signal:
- Dimension 1: $0.01 \cdot 1 = 0.01$
- Dimension 2: $100 \cdot 1 = 100$

The paper's $w^*_{0:1}$ selects dimension 2. The code selects dimension 1. **Wrong.**

However, note that the eigenvalues array is expected to be sorted non-increasingly before being passed in. In the code, `power_law_config` produces eigenvalues already in decreasing order, and `configs.py` passes them directly to the model. So this particular failure mode only triggers if someone passes unsorted eigenvalues or a custom covariance structure.

### Case 3: Partially informative components with unequal weights

Suppose $k=2$ but $w^* = (0, 0, 1, 1, 0, \ldots)$ -- signal is in dimensions 3 and 4, not 1 and 2.

Signal:
- Dimensions 1, 2: $\lambda_j \cdot 0 = 0$
- Dimensions 3, 4: $\lambda_3 (1)^2 = 9^{-1}$, $\lambda_4 (1)^2 = 16^{-1}$

Paper's $w^*_{0:2}$: dimensions 3 and 4.
Code's $w^*_{0:2}$: dimensions 1 and 2 (which are zero -- $w^*_{0:2} = \mathbf{0}$).

This would make $\widehat{\mathcal{L}}(w^*_{0:2}) = \widehat{\mathcal{L}}(\mathbf{0}) = \ln 2 \approx 0.693$, a constant threshold far above what GD reaches quickly. The stopping condition would likely never fire, or fire at iteration $t=1$, completely breaking the experiment.

---

## Summary

| Scenario | Code correct? |
|----------|--------------|
| `power_law_config` (default) | Yes |
| Any config where $w^*_j = c$ for $j \leq k$, $w^*_j = 0$ for $j > k$, and $\lambda_j$ is decreasing | Yes |
| Custom $w^*$ with signal outside the first $k$ components | No |
| Unsorted eigenvalues | No (but this is also a violation of the model assumption) |

## Recommendation

The code should either:

1. **Document the assumption explicitly** -- add a comment or assertion in `run_gd` that warns if the natural ordering does not match the $\pi$ ordering. For example: assert that $\lambda_j (w^*_j)^2$ is non-increasing for the first $k+1$ dimensions.

2. **Implement the general case** -- compute the $\pi$ ordering explicitly and build $w^*_{0:k}$ from it. This would make the stopping condition correct for any $(w^*, \Sigma)$ pair:

```python
signal_per_dim = self.eigenvalues * (self.w_star ** 2)
pi = np.argsort(-signal_per_dim)  # sort descending
w_star_k = np.zeros(self.d)
for i in range(self.k):
    j = pi[i]
    w_star_k += np.dot(self.w_star, np.eye(self.d)[j]) * np.eye(self.d)[j]
# Simplified:
# w_star_k[pi[:self.k]] = self.w_star[pi[:self.k]]
```

Option 2 is straightforward to implement and makes the code match the paper exactly for any config.
