# Proposed Change to `run_gd` Inner Loop

## Math Background: Logistic Regression Loss and Gradient

### The model

We have data $(x_i, y_i)$ for $i = 1, \dots, n$, where $y_i \in \{-1, +1\}$ and $x_i \in \mathbb{R}^d$.

The probability model is:

$$P(y \mid x) = \frac{1}{1 + \exp(-y \cdot x^\top w)}$$

### Empirical logistic loss

$$\hat{\mathcal{L}}(w) = \frac{1}{n} \sum_{i=1}^{n} \ln\!\left(1 + \exp(-y_i \cdot x_i^\top w)\right)$$

Define the **margins** $m_i = y_i \cdot x_i^\top w$. Then:

$$\hat{\mathcal{L}}(w) = \frac{1}{n} \sum_{i=1}^{n} \ln(1 + e^{-m_i})$$

In code: `loss = np.mean(np.logaddexp(0, -margins))` (numerically stable version of the above).

**Why `logaddexp`?** The function `logaddexp(a, b)` computes $\ln(e^a + e^b)$ in a numerically stable way (handles large/small exponents without overflow). Setting $a = 0$ and $b = -m_i$:

$$\text{logaddexp}(0, -m_i) = \ln(e^0 + e^{-m_i}) = \ln(1 + e^{-m_i})$$

This is exactly the per-sample loss. Then `np.mean(...)` gives the $\frac{1}{n}\sum$. Same formula — just written in a way that avoids numerical issues when $m_i$ is very large or very negative.

### Gradient of the loss

Taking the derivative with respect to $w$:

$$\nabla_w \hat{\mathcal{L}}(w) = \frac{1}{n} \sum_{i=1}^{n} \frac{-y_i \cdot x_i}{1 + \exp(m_i)}$$

Define $s_i = \frac{-1}{1 + \exp(m_i)}$ (the "negative sigmoid" of the margin). Then:

$$\nabla_w \hat{\mathcal{L}}(w) = \frac{1}{n} \sum_{i=1}^{n} s_i \cdot y_i \cdot x_i = \frac{1}{n} X^\top (s \odot y)$$

In code:
```python
sigmoid_neg = -1.0 / (1.0 + np.exp(margins))   # s_i for each sample
grad = (self.X.T @ (sigmoid_neg * self.y)) / self.n
```

### Key observation

Both the loss and the gradient depend on the **same margins** $m_i = y_i \cdot x_i^\top w$. The expensive part is computing $X w$ (an $n \times d$ matrix times a $d$-vector). Once we have margins, both loss and gradient are cheap element-wise operations.

---

## Current Code (before change)

```python
for t in range(self.t_current + 1, T + 1):
    grad = self.logistic_gradient(w)        # computes margins = y * (X @ w) internally
    w = w - self.eta * grad

    loss = self.empirical_logistic_loss(w)   # computes margins = y * (X @ w) AGAIN
```

**Problem:** `X @ w` (an n×d matrix times a d-vector) is computed **twice** per iteration — once inside `logistic_gradient` and once inside `empirical_logistic_loss`. They both need the same `margins = y * (X @ w)`.

## Proposed Code

```python
for t in range(self.t_current + 1, T + 1):
    # Step 1: Compute margins ONCE
    margins = self.y * (self.X @ w)

    # Step 2: Gradient from those margins (same math as logistic_gradient)
    sigmoid_neg = -1.0 / (1.0 + np.exp(margins))
    grad = (self.X.T @ (sigmoid_neg * self.y)) / self.n

    # Step 3: Loss from those same margins (same math as empirical_logistic_loss)
    loss = np.mean(np.logaddexp(0, -margins))

    # Step 4: Take the GD step
    w = w - self.eta * grad

    # Step 5: Check stopping condition using the loss we already have
    if prev_loss is not None and loss <= loss_k_truncated <= prev_loss:
        self.stopping_times.append(t)
    prev_loss = loss
```

## What changes and what doesn't

- `logistic_gradient()` and `empirical_logistic_loss()` methods are **untouched** — they're still available for use elsewhere.
- We only inline their math inside the `run_gd` loop to avoid redundant `X @ w` computation.
- The math is identical — no numerical difference.

## Important note on loss timing

The loss computed from `margins` is the loss **before** the gradient step (i.e., loss at w_{t-1}, not w_t). This is because we compute margins from `w` before updating `w`. So `loss` here is actually `L(w_{t-1})`.

This needs to be accounted for in the stopping condition and in what we store. I want to confirm with you how to handle this before proceeding.
