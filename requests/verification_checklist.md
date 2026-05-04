# Verification Checklist

## Order of verification

Work bottom-up: verify the primitives first, then the logic that uses them.

---

### Step 1: Data generation (`model.py`, `generate_data`)

**What to check:**
- X = Z * sqrt(eigenvalues) where Z ~ N(0, I). This gives x ~ N(0, diag(eigenvalues)). Confirm this matches Assumption 1.
- Label generation: logits = X @ w_star, probs = sigmoid(logits), y = 2*Bernoulli(probs) - 1. Confirm this matches Pr(y=+1|x) = 1/(1+exp(-x^T w*)).

**Pay attention to:**
- The sign convention. The paper uses Pr(y|x) = 1/(1+exp(-y x^T w*)). Make sure the code generates y=+1 with probability sigmoid(x^T w*) and y=-1 with probability sigmoid(-x^T w*). These are equivalent — verify.

---

### Step 2: Empirical loss (`model.py`, `empirical_logistic_loss`)

**What to check:**
- L_hat(w) = (1/n) sum_i log(1 + exp(-y_i x_i^T w)).
- The code uses `np.logaddexp(0, -margins)` where margins = y * (X @ w). Confirm logaddexp(0, -m) = log(1 + exp(-m)).

**Pay attention to:**
- This should be straightforward. Just confirm the formula matches the paper's definition.

---

### Step 3: Gradient (`model.py`, inside `run_gd` loop)

**What to check:**
- The gradient of L_hat(w) = (1/n) sum log(1 + exp(-y_i x_i^T w)).
- Derivative: dL/dw = (1/n) sum [ -y_i x_i * sigmoid(-y_i x_i^T w) ] = (1/n) sum [ -y_i x_i / (1 + exp(y_i x_i^T w)) ].
- Code computes: `sigmoid_neg = -1/(1+exp(margins))`, then `grad = (X^T @ (sigmoid_neg * y)) / n`.
- Expand: grad_j = (1/n) sum_i [ x_{ij} * (-1/(1+exp(y_i x_i^T w))) * y_i ]. Confirm this equals the derivative above.

**Pay attention to:**
- Sign correctness. A sign error here means GD goes uphill.
- The standalone `logistic_gradient` method should give the same result as the inlined version in `run_gd`. (Could write a quick test.)

---

### Step 4: The GD loop timing / what gets recorded (`model.py`, `run_gd`)

**What to check:**
- At iteration t in the loop, what is `w` at the start? What is `loss`? What is `w` after the step?
- What gets appended to `w_history` and `loss_history` at checkpoint t?

**Pay attention to:**
- **This is where the off-by-one lives.** The loss recorded at time t is computed BEFORE the step (so it's L(w_{t-1})), but w_history records w AFTER the step (so it's w_t). They don't correspond to the same iterate.
- Population loss is evaluated on the post-step w (i.e., w_t), so it's consistent with w_history but inconsistent with loss_history.
- Decide: is this a bug that matters for Dashboard 3, or just a labeling shift?

---

### Step 5: Early stopping condition (`model.py`, `run_gd`, lines 142-144)

**What to check:**
- Paper's Theorem 3.1 says: there exists t such that L_hat(w_t) <= L_hat(w*_{0:k}) <= L_hat(w_{t-1}).
- Code checks: `loss <= loss_k_truncated <= prev_loss` and records `t-1`.
- Given the timing from Step 4: `loss` = L(w_{t-1}), `prev_loss` = L(w_{t-2}).
- So the code is checking: L(w_{t-1}) <= L(w*_{0:k}) <= L(w_{t-2}), and recording t-1.

**Pay attention to:**
- Does this correctly identify the iterate whose loss first crosses below L(w*_{0:k})? Trace it carefully with the off-by-one from Step 4.
- The paper says the stopping time t satisfies L_hat(w_t) <= ref <= L_hat(w_{t-1}). If the code's "loss" variable is really L(w_{t-1}), then the code is finding the t-1 such that L(w_{t-1}) <= ref, and the previous step had L(w_{t-2}) >= ref. That means w_{t-1} is the first iterate below the reference. Recording t-1 seems correct.

---

### Step 6: w*_{0:k} definition (`model.py`, `run_gd`, lines 124-126)

**What to check:**
- Paper defines w*_{0:k} using a reordering pi(i) such that lambda_{pi(i)} (u_{pi(i)}^T w*)^2 is decreasing. Then w*_{0:k} = sum_{i<=k} u_{pi(i)} u_{pi(i)}^T w*.
- Code does: `w_star_k[:k] = w_star[:k]` (takes first k components).

**Pay attention to:**
- This is only correct when the eigenvalue ordering already matches the pi(i) ordering. With eigenvalues = i^{-2} and w* = [1,...,1,0,...,0], the quantity lambda_i (u_i^T w*)^2 = i^{-2} * 1 for i<=k and 0 for i>k. So pi(i) = i for i<=k, and the code is correct for this specific config.
- **If you ever change w* or eigenvalues to something non-standard, this line breaks.** Decide if you care (probably fine for now since all experiments use the default config).

---

### Step 7: Population loss (`model.py`, `population_logistic_loss`)

**What to check:**
- Same formula as empirical loss but with fresh MC samples from the true distribution.
- Uses fixed seed=999 for every evaluation.

**Pay attention to:**
- Fixed seed means the same synthetic population is used every time. Good for smooth curves within a run, but means all M runs in Dashboard 3 evaluate pop loss on the exact same population sample. This is probably fine (reduces noise), but be aware.
- The MC sample size is `pop_samples_per_dim * d`. For d=500 and default 25, that's 12,500 samples. Is that enough? You could check by running it twice with different seeds and comparing.

---

### Step 8: Step size (`configs.py`, `theoretical_eta`)

**What to check:**
- Paper: eta <= 1 / (C0 (1 + tr(Sigma) + lambda_1 ln(1/delta) / n)).
- Code: `1.0 / (C0 * (1 + tr_sigma + lambda_1 * np.log(1.0/delta) / n))` with C0=2.0, delta=0.01.

**Pay attention to:**
- This is a direct transcription. Just confirm the parentheses match the paper.

---

### Step 9: Max-margin direction (`model.py`, `compute_max_margin_direction`)

**What to check:**
- SVM dual: minimize (1/2) alpha^T (Y G Y) alpha - 1^T alpha, subject to alpha >= 0, where G = X X^T and Y = diag(y).
- Then w_svm = X^T (alpha * y), normalized to unit vector.

**Pay attention to:**
- Is this the correct dual for max-margin? The primal is: max_{||w||=1} min_i y_i x_i^T w. The dual should give the same solution.
- The L-BFGS-B solver has tolerances (ftol=1e-15, gtol=1e-12). For Dashboard 3 where you average over runs, small solver inaccuracies could matter. Maybe check convergence quality.

---

### Step 10: Resumability (`configs.py`, `run_and_save` + `run_gd` continuation)

**What to check:**
- `run_and_save` stores w_history, loss_history, etc. but NOT X, y.
- To resume: you'd re-create the model with same seed, call `generate_data()` (regenerates same X, y), then somehow restore the GD state and call `run_gd(T_new)`.

**Pay attention to:**
- There's no `resume_from_pkl` function in the current code (despite configs.py mentioning it in PROJECT_STATE.md). How does resumption actually work? Is it just: re-create model, generate data, set `model.w_current = last w from history`, set `model.t_current`, and call `run_gd(T_new)`? If so, you need to also restore loss_history, w_history, stopping_times, etc. Verify this path works before running at scale.
