# Conjecture 1

I want to use the basic setup for eta and n and d and set T=10,000 for now. In the overparameterized regime we have d >> n.


**Data Model.** We mainly focus on a *well-specified* setting formalized by the following conditions. However, part of our results can also be applied to misspecified cases.

**Assumption 1** (Well-specification). *Let* $\Sigma \in \mathbb{H}^{\otimes 2}$ *be positive semi-definite (PSD) and* $\text{tr}(\Sigma) < \infty$. *Let* $\mathbf{w}^* \in \mathbb{H}$ *be such that* $\|\mathbf{w}^\*\|_\Sigma < \infty$. *Assume that* $(\mathbf{x}, y) \in \mathbb{H} \otimes \{\pm 1\}$ *is given by*

$$\mathbf{x} \sim \mathcal{N}(0, \Sigma), \quad \Pr(y | \mathbf{x}) = \frac{1}{1 + \exp(-y \mathbf{x}^\top \mathbf{w}^*)}$$

Generate a fixed dataset of n points as explained above for our experiment.

Run gradient descent for T rounds and store $w_1, \ldots, w_T$, and
the goal is to see whether $\frac{w_t}{\|w_t\|}$
converges to $\tilde{w}$ or $w^*$.

The goal is to plot but I don't know what the best plot is when we want to compare vectors in high dimensional space.

---

## Main Question

The main question we care about in this conjecture is:
**Is $\tilde{w}$ on expectation an unbiased estimator of $w^*$?**

That is, given M independent training sets of size n, each yielding a
max-margin direction $\tilde{w}_m$, does

$$\frac{1}{M} \sum_{m=1}^{M} \tilde{w}_m \to \frac{w^\*}{\|w^\*\|}$$

as $M \to \infty$?

### Claude's insight notes:

By symmetry, the uninformative dimensions (k+1 to d) of each $\tilde{w}_m$
are driven purely by noise in the training set. Since there is no signal in
those dimensions, their contributions should cancel out when averaging over
independent samples. So $\mathbb{E}[\tilde{w}]$ should live entirely in the
informative subspace (the first k components).

However, even within the informative subspace, it is not obvious that
$\mathbb{E}[\tilde{w}]$ points in the direction of $w^\*$. The eigenvalues
decay as $\lambda_i = i^{-2}$, which means each informative dimension has a
different signal-to-noise ratio. Dimensions with larger eigenvalues (e.g.
i=1) have stronger signal relative to noise, while dimensions with smaller
eigenvalues (e.g. i=100) have weaker signal. The max-margin solution is
determined by support vectors, and the geometry of support vectors could be
biased toward dimensions with more variance (larger eigenvalues). This means
$\mathbb{E}[\tilde{w}]$ might overweight some informative dimensions and
underweight others, resulting in a direction that lives in the right subspace
but is not proportional to $w^\*$.

If it turns out that $\mathbb{E}[\tilde{w}]$ does converge to the direction
of $w^\*$, that would be a strong result: it would mean $\tilde{w}$ is an
unbiased estimator of the right direction, just an extremely noisy one.

-- Claude (Opus 4.6)
