# Experiment Ideas

## Question 1: Population Gradient Descent vs Empirical Gradient Descent

What happens if instead of empirical GD (which uses n=1000 training points to
compute the gradient), we use the true population gradient? Does the algorithm
still exhibit the same behavior -- early iterates aligning with w* and then
drifting toward w_tilde asymptotically?

The population gradient can be approximated via Monte Carlo by averaging the
gradient over a very large number of fresh samples at each step (or
analytically if tractable).

Key sub-questions:
- Does the early stopping phenomenon still appear with population GD?
- Does the asymptotic direction still converge to the max-margin direction
  w_tilde, or does it stay aligned with w*?
- If population GD does not drift away from w*, this would confirm that the
  drift toward w_tilde is purely an artifact of overfitting the finite
  training set.

## Question 2: Does the Expected Max-Margin Direction Converge to w*?

Given M independent training sets of size n, each yielding a max-margin
direction w_tilde_m, does the average (1/M) sum_{m=1}^{M} w_tilde_m converge
in direction to w* as M grows?

In other words, is E[w_tilde] proportional to w*?

Key sub-questions:
- For each independent sample, the uninformative dimensions (k+1 to d) of
  w_tilde are driven by noise. By symmetry, these should cancel out in
  expectation. Does the average w_tilde live entirely in the informative
  subspace (first k components)?
- Even if the average lives in the informative subspace, does it point in the
  direction of w* specifically? The eigenvalues decay as i^{-2}, so different
  informative dimensions have different signal-to-noise ratios, which could
  bias the max-margin solution to favor some dimensions over others.
- If E[w_tilde] does converge to w*, it would mean w_tilde is an unbiased
  estimator of the right direction, just a very noisy one.

## Question 3 (Main Goal): Data-Dependent Early Stopping

The paper's early stopping result is theoretical -- the optimal stopping time
depends on unknown population quantities (w*, the eigenvalue structure, etc.)
that a practitioner does not have access to. The big question is: can we
design a data-dependent stopping rule that achieves similar benefits using
only observable quantities from the training process?

This is the central practical question. A good data-dependent rule would make
the theoretical guarantees of early stopping actually usable.

Possible directions to explore:
- Held-out validation: split the n samples, train on one part, monitor
  population-like loss on the held-out part. Simple but wastes data, which
  matters when n is already small relative to d.
- Observable training statistics: can we detect the "turning point" (where
  the iterate starts drifting from w* toward w_tilde) by monitoring
  quantities we can compute, such as:
  - Rate of change of the iterate direction (||w_t/||w_t|| - w_{t-1}/||w_{t-1}||||)
  - Gradient norm or its rate of decay
  - Empirical loss curvature or second-order information
  - Change in the margin distribution over training points
- Stability-based criteria: run GD on bootstrap resamples or slight
  perturbations of the data, and stop when the iterates start to diverge
  across perturbations (indicating the algorithm is starting to fit noise).
- Connection to Questions 1 and 2: understanding the population GD behavior
  and the structure of w_tilde may give clues about what observable
  signatures mark the transition from "learning signal" to "fitting noise."
