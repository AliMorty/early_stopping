# Code Review: model.py

**File:** `experiment/model.py`
**Class:** `OverparameterizedLogisticRegression`

---

## 1. Data Generation (`generate_data`)

The paper's data model (Assumption 1) requires:

$$\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}), \quad \Pr(y \mid \mathbf{x}) = \frac{1}{1 + \exp(-y \mathbf{x}^\top \mathbf{w}^*)}$$

where $\boldsymbol{\Sigma} = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$ (we work in the eigenbasis, so $U = I$).

**Implementation:**
```python
Z = rng.randn(self.n, self.d)
self.X = Z * np.sqrt(self.eigenvalues)[np.newaxis, :]
```
Sampling $Z \sim \mathcal{N}(0, I)$ and scaling each column $j$ by $\sqrt{\lambda_j}$ gives rows distributed as $\mathcal{N}(0, \Sigma)$. **Correct.**

```python
logits = self.X @ self.w_star
probs = 1.0 / (1.0 + np.exp(-logits))
self.y = 2.0 * (rng.rand(self.n) < probs).astype(float) - 1.0
```
This correctly implements $\Pr(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \mathbf{w}^*)$ and maps labels to $\{+1, -1\}$. **Correct.**

---

## 2. Loss and Gradient

**Empirical logistic risk:**

$$\widehat{\mathcal{L}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ln(1 + e^{-y_i \mathbf{x}_i^\top \mathbf{w}})$$

```python
margins = self.y * (self.X @ w)
return np.mean(np.logaddexp(0, -margins))
```
`np.logaddexp(0, -m)` computes $\ln(e^0 + e^{-m}) = \ln(1 + e^{-m})$. Numerically stable. **Correct.**

**Gradient of the empirical risk:**

$$\nabla \widehat{\mathcal{L}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \frac{-y_i \mathbf{x}_i}{1 + e^{y_i \mathbf{x}_i^\top \mathbf{w}}}$$

```python
margins = self.y * (self.X @ w)
sigmoid_neg = -1.0 / (1.0 + np.exp(margins))
return (self.X.T @ (sigmoid_neg * self.y)) / self.n
```
`sigmoid_neg[i]` $= -1/(1 + e^{y_i x_i^\top w})$, so `sigmoid_neg * self.y` $= -y_i/(1 + e^{y_i x_i^\top w})$. The outer product with $X^\top$ and division by $n$ gives the gradient exactly. **Correct.**

---

## 3. GD Loop and Shared Margins Optimization

Inside `run_gd`, margins are computed once per iteration and reused for both the gradient and the loss:

```python
margins = self.y * (self.X @ w)
sigmoid_neg = -1.0 / (1.0 + np.exp(margins))
grad = (self.X.T @ (sigmoid_neg * self.y)) / self.n
loss = np.mean(np.logaddexp(0, -margins))  # L(w_{t-1}), before step
w = w - self.eta * grad
```

This is a valid optimization: the gradient and the loss both depend only on the margins $y_i x_i^\top w$. **Correct and efficient.**

---

## 4. Early Stopping Condition

Theorem 3.1 defines the stopping time as the first $t$ satisfying:

$$\widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1})$$

The implementation checks this inside the GD loop at iteration $t$:

```python
loss = np.mean(np.logaddexp(0, -margins))  # = L_hat(w_{t-1}), before step
...
w = w - self.eta * grad                    # w is now w_t
# Check: L(w_{t-1}) <= L(w*_0:k) <= L(w_{t-2})
if prev_loss is not None and loss <= loss_k_truncated <= prev_loss:
    self.stopping_times.append(t - 1)
```

At loop iteration $t$, `loss` $= \widehat{\mathcal{L}}(w_{t-1})$ and `prev_loss` $= \widehat{\mathcal{L}}(w_{t-2})$. The condition fires when $\widehat{\mathcal{L}}(w_{t-1}) \leq \widehat{\mathcal{L}}(w^*_{0:k}) \leq \widehat{\mathcal{L}}(w_{t-2})$, and stores $t-1$. Substituting $\tau = t-1$:

$$\widehat{\mathcal{L}}(\mathbf{w}_\tau) \leq \widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}) \leq \widehat{\mathcal{L}}(\mathbf{w}_{\tau-1})$$

This matches Theorem 3.1 exactly. **Correct.**

The k-truncated reference vector is computed as:
```python
w_star_k = np.zeros(self.d)
w_star_k[:self.k] = self.w_star[:self.k]
```
This is $w^*_{0:k}$ using the natural component ordering. See the note in section 7 about when this is valid.

---

## 5. Loss Stored in `loss_history` (Minor Inconsistency)

The `loss_history` appends `(t, loss)` at each checkpoint, where `loss` is $\widehat{\mathcal{L}}(w_{t-1})$ -- the loss computed **before** the GD step at iteration $t$. But `w_history` appends `(t, w.copy())` where `w` is $w_t$ -- the iterate **after** the step.

This means `loss_history[i]` stores $\widehat{\mathcal{L}}(w_{t_i - 1})$, not $\widehat{\mathcal{L}}(w_{t_i})$. The two histories are misaligned by one step.

**Impact:** For plotting purposes (large $T$), this is negligible. For the stopping time detection, `prev_loss` is maintained independently and correctly. The inconsistency is low-risk but worth knowing if you ever cross-reference `w_history` and `loss_history` directly.

---

## 6. Max-Margin Direction (`compute_max_margin_direction`)

The method solves the SVM dual problem:

$$\max_{\boldsymbol{\alpha} \geq 0} \sum_i \alpha_i - \frac{1}{2} \boldsymbol{\alpha}^\top (Y X X^\top Y) \boldsymbol{\alpha}$$

and recovers the primal solution via $\tilde{w} = X^\top (\alpha^* \odot y)$, then normalizes.

```python
G = self.X @ self.X.T
YGY = np.outer(self.y, self.y) * G
...
w_svm = self.X.T @ (alpha_star * self.y)
self.w_tilde = w_svm / norm(w_svm)
```

The dual formulation is standard (KKT conditions recover $w = X^\top (\alpha \odot y)$). **Correct.**

The objective sign is correct: `dual_objective` minimizes $\frac{1}{2} \alpha^\top YGY \alpha - \mathbf{1}^\top \alpha$, which is the negation of the max objective. **Correct.**

---

## 7. Suggested Tests

- **Gradient check:** Verify `logistic_gradient` numerically using finite differences on a small example (e.g., $n=5$, $d=3$). Run: `scipy.optimize.check_grad(model.empirical_logistic_loss, model.logistic_gradient, w_test)` and confirm the result is near zero.

- **Stopping time check:** On a small run, manually compute $\widehat{\mathcal{L}}(w^*_{0:k})$ and verify that the iteration stored in `stopping_times[0]` satisfies the two-sided sandwich condition by checking `loss_history` around that iteration.

- **Loss/w_history alignment:** Confirm the off-by-one: compute `model.empirical_logistic_loss(w_history[i][1])` for a few $i$ and compare against `loss_history[i][1]`. You should find they differ by approximately one GD step.

- **w_star_k validity for custom configs:** If you use a custom $w^*$ where informative components are not in the first $k$ positions, `w_star_k[:k] = w_star[:k]` will be wrong. The correct implementation should sort by $\lambda_i (u_i^\top w^*)^2$ as defined in Section 2 of the paper. For the default `power_law_config` this is not an issue.
