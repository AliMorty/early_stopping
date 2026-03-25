# Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression

**Jingfeng Wu¹ · Peter L. Bartlett\*¹² · Matus Telgarsky\*³ · Bin Yu\*¹**

\*Equal contribution. ¹University of California, Berkeley. ²Google DeepMind. ³New York University.

*Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025.*

arXiv: 2502.13283v2 [cs.LG] 30 Jun 2025

---

## Abstract

In overparameterized logistic regression, gradient descent (GD) iterates diverge in norm while converging in direction to the maximum $\ell_2$-margin solution—a phenomenon known as the implicit bias of GD. This work investigates additional regularization effects induced by early stopping in well-specified high-dimensional logistic regression. We first demonstrate that the excess logistic risk vanishes for early-stopped GD but diverges to infinity for GD iterates at convergence. This suggests that early-stopped GD is well-calibrated, whereas asymptotic GD is statistically inconsistent. Second, we show that to attain a small excess zero-one risk, polynomially many samples are sufficient for early-stopped GD, while exponentially many samples are necessary for any interpolating estimator, including asymptotic GD. This separation underscores the statistical benefits of early stopping in the overparameterized regime. Finally, we establish nonasymptotic bounds on the norm and angular differences between early-stopped GD and $\ell_2$-regularized empirical risk minimizer, thereby connecting the implicit regularization of GD with explicit $\ell_2$-regularization.

---

## 1. Introduction

Modern machine learning often operates in the *overparameterized* regime, where the number of parameters exceeds the number of training data. Despite this, models trained by *gradient descent* (GD) often generalize well even in the absence of explicit regularization (Zhang et al., 2021; Neyshabur et al., 2017; Bartlett et al., 2021). The common explanation is that GD exhibits certain *implicit regularization* effects that prevent overfitting.

The implicit regularization of GD is relatively well understood in regression settings. Using overparameterized linear regression as an example, amongst all interpolators, GD asymptotically converges to the minimum $\ell_2$-norm interpolator (Zhang et al., 2021). Moreover, when the data covariance satisfies certain conditions, the minimum $\ell_2$-norm interpolator achieves vanishing excess risk while fitting training data with *constant* amount of noise, a phenomenon known as *benign overfitting* (see Bartlett et al., 2020; Tsigler & Bartlett, 2023, and references therein). When the data covariance is general, although benign overfitting may not occur, early-stopped GD (and one-pass stochastic GD) can still achieve vanishing excess risk (Bühlmann & Yu, 2003; Yao et al., 2007; Lin & Rosasco, 2017; Dieuleveut & Bach, 2016; Zou et al., 2023; 2022; Wu et al., 2022a). This suggests early stopping provides an additional regularization effect for GD in linear regression. Moreover, the statistical effects of early stopping are known to be comparable to that of $\ell_2$-regularization in linear regression (Suggala et al., 2018; Ali et al., 2019; Zou et al., 2021; Sonthalia et al., 2024).

However, the picture is less complete for classification, where the risk is measured by the logistic loss and the zero-one loss instead of the squared loss. In overparameterized logistic regression, GD diverges in norm while converging in direction to the maximum $\ell_2$-margin solution (see Soudry et al., 2018; Ji & Telgarsky, 2018, and Proposition 2.2 in Section 2), which is in contrast with GD's convergence to the (bounded!) minimum $\ell_2$-norm solution in the linear regression setting. In standard (finite-dimensional, low-noise, large margin) classification settings, the asymptotic implicit bias of GD implies generalization via classical margin theory (Bartlett & Shawe-Taylor, 1999). More recently, certain high-dimensional settings exhibit well-behaved maximum margin solutions and benign overfitting (see Montanari et al., 2019, for example), but it is unclear if these results apply more broadly or represent special cases. Moreover, if the maximum $\ell_2$-margin solution generalizes poorly, new techniques are required, as the aforementioned least squares techniques cannot be easily adapted owing to their heavy dependence upon the explicit linear algebraic form of GD's path specific to least squares.

**Contributions.** This work investigates the beneficial regularization effects of early stopping in GD for overparameterized logistic regression. We focus on a well-specified setting where the feature vector follows an anisotropic Gaussian design and the binary label conditional on the feature is given by a logistic model (see Assumption 1 in Section 2). We are particularly interested in the regime where the label contains a constant level of noise. We establish the following results.

1. **Calibration via early stopping.** We first derive risk upper bounds for early-stopped GD that can be applied in the overparameterized regime. With an oracle-chosen stopping time, early-stopped GD achieves vanishing excess logistic risk and excess zero-one error (as the sample size grows) for every well-specified logistic regression problem. Furthermore, its naturally induced conditional probability approaches the true underlying conditional probability model. These properties suggest that early-stopped GD is *consistent* and *calibrated* for every well-specified logistic regression problem, even in the overparameterized regime.

2. **Advantages over interpolation.** We then provide negative results for GD without early stopping. We show that GD at convergence, in contrast to the typical successes of maximum margin predictors, suffers from an *unbounded* logistic risk and a *constant* calibration error in the overparameterized regime. Moreover, for a broad class of overparameterized logistic regression problems, to attain a small excess zero-one error, early-stopped GD only needs *polynomially* many samples, whereas any interpolating estimators, including asymptotic GD, requires at least *exponentially* many samples. These results underscore the statistical benefits of early stopping.

3. **Connections to $\ell_2$-regularization.** Finally, we compare the GD path (formed by GD iterates with all possible stopping times) with the $\ell_2$-regularization path (formed by $\ell_2$-regularized empirical risk minimizers with all possible regularization strengths). For general convex and smooth problems, including logistic regression, these two paths differ in norm by a factor between 0.585 and 3.415, and differ in direction by an angle no more than $\pi/4$. Specific to overparameterized logistic regression, the $\ell_2$-distance of the two paths is asymptotically zero in a widely considered situation but may diverge to infinity in the worst case. These findings partially explain the implicit regularization of early stopping via its connections with the explicit $\ell_2$-regularization.

**Notation.** For two positive-valued functions $f(x)$ and $g(x)$, we write $f(x) \lesssim g(x)$ or $f(x) \gtrsim g(x)$ if there exists a constant $c > 0$ such that $f(x) \leq cg(x)$ or $f(x) \geq cg(x)$ for every $x$, respectively. We write $f(x) \asymp g(x)$ if $f(x) \lesssim g(x) \lesssim f(x)$. We use the standard big-O notation. For two vectors $\mathbf{u}$ and $\mathbf{v}$ in a Hilbert space, we denote their inner product by $\langle \mathbf{u}, \mathbf{v} \rangle$ or equivalently, $\mathbf{u}^\top \mathbf{v}$. For two matrices $\mathbf{A}$ and $\mathbf{B}$ of appropriate dimension, we define their inner product as $\langle \mathbf{A}, \mathbf{B} \rangle := \mathrm{tr}(\mathbf{A}^\top \mathbf{B})$. For a positive semi-definite (PSD) matrix $\mathbf{A}$ and a vector $\mathbf{v}$ of appropriate dimension, we write $\|\mathbf{v}\|_\mathbf{A}^2 := \mathbf{v}^\top \mathbf{A} \mathbf{v}$. In particular, we write $\|\mathbf{v}\| := \|\mathbf{v}\|_\mathbf{I}$. For a positive integer $n$, we write $[n] := \{1, \ldots, n\}$.

---

## 2. Preliminaries

Let $(\mathbf{x}, y) \in \mathbb{H} \otimes \{\pm 1\}$ be a pair of features and the corresponding binary label sampled from an unknown population distribution. Here $\mathbb{H}$ is a finite or countably infinite dimensional Hilbert space. For a parameter $\mathbf{w} \in \mathbb{H}$, define its population **logistic risk** as

$$\mathcal{L}(\mathbf{w}) := \mathbb{E}\,\ell(y\mathbf{x}^\top\mathbf{w}), \quad \text{where } \ell(t) := \ln(1 + e^{-t}),$$

and define its population **zero-one error** as

$$\mathcal{E}(\mathbf{w}) := \mathbb{E}\,\mathbf{1}\!\left[y\mathbf{x}^\top\mathbf{w} \leq 0\right] = \Pr(y\mathbf{x}^\top\mathbf{w} \leq 0),$$

where the expectation is over the population distribution of $(\mathbf{x}, y)$. It is worth noting that, different from the logistic risk $\mathcal{L}(\mathbf{w})$, the zero-one error $\mathcal{E}(\mathbf{w})$ is insensitive to the parameter norm. Moreover, we measure the **calibration error** of a parameter $\mathbf{w} \in \mathbb{H}$ by

$$\mathcal{C}(\mathbf{w}) := \mathbb{E}\left|p(\mathbf{w}; \mathbf{x}) - \Pr(y = 1 | \mathbf{x})\right|^2,$$

where $p(\mathbf{w}; \mathbf{x})$ is a naturally induced conditional probability given by

$$p(\mathbf{w}; \mathbf{x}) := \frac{1}{1 + \exp(-\mathbf{x}^\top \mathbf{w})}.$$

We say an estimator $\hat{\mathbf{w}}$ is *consistent* (for classification) if it attains the Bayes zero-one error asymptotically, that is, $\mathcal{E}(\hat{\mathbf{w}}) - \min \mathcal{E} \to 0$. We say an estimator $\hat{\mathbf{w}}$ is *calibrated* if its induced conditional probability predicts the true one asymptotically (Foster & Vohra, 1998), that is, $\mathcal{C}(\hat{\mathbf{w}}) \to 0$.

**Gradient descent.** Let $(\mathbf{x}_i, y_i)_{i=1}^n$ be $n$ independent copies of $(\mathbf{x}, y)$. Define the empirical risk as

$$\widehat{\mathcal{L}}(\mathbf{w}) := \frac{1}{n} \sum_{i=1}^n \ell(y_i \mathbf{x}_i^\top \mathbf{w}), \quad \mathbf{w} \in \mathbb{H}.$$

Then the iterates of *gradient descent* (GD) are given by

$$\mathbf{w}_0 = \mathbf{0}, \quad \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \widehat{\mathcal{L}}(\mathbf{w}_t), \quad t \geq 0, \tag{GD}$$

where $\eta > 0$ is a fixed stepsize. We consider zero initialization to simplify the presentation, which does not cause the loss of generality. We aim to compare asymptotic GD, that is, $\mathbf{w}_\infty$, with early-stopped GD, that is, $\mathbf{w}_t$ at a certain finite stopping time $t < \infty$.

**Data model.** We mainly focus on a well-specified setting formalized by the following conditions. However, part of our results can also be applied to misspecified cases.

> **Assumption 1 (Well-specification).** *Let $\boldsymbol{\Sigma} \in \mathbb{H}^{\otimes 2}$ be positive semi-definite (PSD) and $\mathrm{tr}(\boldsymbol{\Sigma}) < \infty$. Let $\mathbf{w}^* \in \mathbb{H}$ be such that $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} < \infty$. Assume that $(\mathbf{x}, y) \in \mathbb{H} \otimes \{\pm 1\}$ is given by*
> $$\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}), \quad \Pr(y|\mathbf{x}) = \frac{1}{1 + \exp(-y\mathbf{x}^\top\mathbf{w}^*)}.$$

Under this data model, we have the following standard properties for the logistic risk, zero-one error, and calibration error.

> **Proposition 2.1 (Basic properties).** *Under Assumption 1, we have*
>
> *A. $\mathbf{w}^* = \arg\min \mathcal{L}(\cdot)$ and $\mathbf{w}^* \in \arg\min \mathcal{E}(\cdot)$;*
>
> *B. for every $\mathbf{w} \in \mathbb{H}$, it holds that*
> $$\mathcal{E}(\mathbf{w}) - \min\mathcal{E} \leq 2\sqrt{\mathcal{C}(\mathbf{w})} \leq \sqrt{2} \cdot \sqrt{\mathcal{L}(\mathbf{w}) - \min\mathcal{L}};$$
>
> *C. if additionally we have $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} \lesssim 1$, then*
> $$\min\mathcal{L} \gtrsim 1, \quad \min\mathcal{E} \gtrsim 1.$$

Proposition 2.1 suggests that the Bayes logistic risk and Bayes zero-one error are attained by the true model parameter $\mathbf{w}^*$. Moreover, the excess zero-one error is controlled by the calibration error, which is further controlled by the excess logistic risk. Thus under Assumption 1, a calibrated estimator is also consistent for classification, and an estimator is calibrated if it attains the Bayes logistic risk asymptotically. However, the reverse might not be true. As we will show later, for overparameterized logistic regression, early-stopped GD is calibrated and consistent for both logistic risk and zero-one error. In contrast, asymptotic GD is poorly calibrated and attains an unbounded logistic risk, although it could be consistent for zero-one error.

**Noise and overparameterization.** Most of our results should be interpreted in the *noisy* and *overparameterized* regime. Specifically, this means

$$\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} \lesssim 1 \quad \text{and} \quad \mathrm{rank}(\boldsymbol{\Sigma}) \geq n.$$

The first condition ensures the population distribution carries a constant amount of noise, as the Bayes logistic risk and Bayes zero-one error are lower bounded by a constant according to Proposition 2.1. In other words, the population distribution is strictly *not* linearly separable. Despite so, the second condition ensures the *linear separability* of the training data almost surely, as the number of effective parameters exceeds the number of training data. In this regime, estimators can *interpolate* the training data, yet this interpolation inherently carries the risk of *overfitting* and *poor calibration*. Our setting aligns well with the prior setting for studying benign overfitting in linear regression (Bartlett et al., 2020; Tsigler & Bartlett, 2023).

**Asymptotic implicit bias.** When the training data is linearly separable (implied by overparameterization), prior works show that GD diverges to infinity in norm while converging in direction to the maximum $\ell_2$-margin direction (Soudry et al., 2018; Ji & Telgarsky, 2018). This characterizes the asymptotic implicit bias of GD. See the following proposition for a precise statement.

> **Proposition 2.2 (Asymptotic implicit bias).** *Assume that $\mathrm{rank}(\mathbf{x}_1, \ldots, \mathbf{x}_n) \geq n$. Then the training data $(\mathbf{x}_i, y_i)_{i=1}^n$ is linearly separable, that is,*
> $$\max_{\|\mathbf{w}\|=1} \min_{i \in [n]} y_i \mathbf{x}_i^\top \mathbf{w} > 0.$$
> *Let $\tilde{\mathbf{w}}$ be the maximum $\ell_2$-margin direction, that is,*
> $$\tilde{\mathbf{w}} := \arg\max_{\|\mathbf{w}\|=1} \min_{i \in [n]} y_i \mathbf{x}_i^\top \mathbf{w}.$$
> *Then $\tilde{\mathbf{w}}$ is unique and the following holds for* (GD) *with any stepsize $\eta > 0$:*
> $$\|\mathbf{w}_t\| \to \infty, \quad \frac{\mathbf{w}_t}{\|\mathbf{w}_t\|} \to \tilde{\mathbf{w}}.$$

**Additional notation.** The following notations are handy for presenting our results. Let $(\lambda_i)_{i \geq 1}$ be the eigenvalues of the data covariance $\boldsymbol{\Sigma}$, sorted in non-increasing order. Let $\mathbf{u}_i$ be the eigenvector of $\boldsymbol{\Sigma}$ corresponding to $\lambda_i$. Let $(\pi(i))_{i \geq 1}$ be resorted indexes such that $\lambda_{\pi(i)}(\mathbf{u}_{\pi(i)}^\top \mathbf{w}^*)^2$ is non-increasing as a function of $i$. Define

$$\mathbf{w}^*_{0:k} := \sum_{i \leq k} \mathbf{u}_{\pi(i)} \mathbf{u}_{\pi(i)}^\top \mathbf{w}^*, \quad \mathbf{w}^*_{k:\infty} := \sum_{i > k} \mathbf{u}_{\pi(i)} \mathbf{u}_{\pi(i)}^\top \mathbf{w}^*.$$

It is clear that $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} < \infty$ implies that $\|\mathbf{w}^*_{k:\infty}\|_{\boldsymbol{\Sigma}} = o(1)$ as $k$ increases.

---

## 3. Upper Bounds for Early-Stopped GD

In this section, we present two risk bounds for early-stopped GD for overparameterized logistic regression and a characterization of the implicit bias of early stopping in GD.

### 3.1. A Bias-Dominating Bound

We first provide a bias-dominating excess logistic risk bound for early-stopped GD in overparameterized logistic regression.

> **Theorem 3.1 (A "bias-dominating" risk bound).** *Suppose that Assumption 1 holds. Let $k$ be an arbitrary index. Suppose that the stepsize for* (GD) *satisfies*
> $$\eta \leq \frac{1}{C_0\left(1 + \mathrm{tr}(\boldsymbol{\Sigma}) + \lambda_1 \ln(1/\delta)/n\right)},$$
> *where $C_0 > 1$ is a universal constant. Then with probability at least $1 - \delta$, there exists a stopping time $t$ such that*
> $$\widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1}).$$
> *Moreover, for* (GD) *with this stopping time we have*
> $$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} \lesssim \sqrt{\frac{\max\left\{1,\, \mathrm{tr}(\boldsymbol{\Sigma})\|\mathbf{w}^*_{0:k}\|^2\right\} \ln^2(n/\delta)}{n}} + \|\mathbf{w}^*_{k:\infty}\|_{\boldsymbol{\Sigma}}^2.$$

The existence of the desired stopping time is because GD minimizes the empirical risk monotonically (Ji & Telgarsky, 2018). In Theorem 3.1, we choose $k$ to minimize the upper bounds. Intuitively, $k$ determines the number of dimensions in which early-stopped GD is able to learn the true parameter. Moreover, early-stopped GD ignores the remaining dimensions and pays an "approximation" error.

**Calibration and consistency.** Theorem 3.1 implies that early-stopped GD attains the Bayes logistic risk asymptotically for any logistic regression problem satisfying Assumption 1. To see this, we pick $k$ as an increasing function of $n$ such that $\|\mathbf{w}^*_{0:k}\| = o(n)$. Then $\|\mathbf{w}^*_{k:\infty}\|_{\boldsymbol{\Sigma}} = o(1)$ since $k$ increases as $n$ increases (recall that $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}$ is finite by Assumption 1). Hence the risk bound in Theorem 3.1 implies that

$$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} = o(1) \quad \text{as } n \text{ increases.}$$

By Proposition 2.1, this also ensures that early-stopped GD induces a conditional probability that approaches the true one and achieves a vanishing excess zero-one error. Hence early-stopped GD is calibrated and consistent for any well-specified logistic regression problem.

As a concrete example, let us consider the following source and capacity conditions (Caponnetto & De Vito, 2007),

$$\lambda_i \asymp i^{-a}, \quad \lambda_i(\mathbf{u}_i^\top \mathbf{w}^*_i)^2 \asymp i^{-b}, \quad a, b > 1. \tag{1}$$

Then Theorem 3.1 implies

$$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} = \begin{cases} \tilde{\mathcal{O}}\!\left(n^{-1/2}\right) & b > a+1, \\ \tilde{\mathcal{O}}\!\left(n^{\frac{1-b}{a+b-1}}\right) & b \leq a+1. \end{cases}$$

This provides an explicit rate on the excess risk. Note that the obtained rate might not be the sharpest. An improved rate under stronger conditions is provided later in Theorem 3.2.

**Stopping time.** Note that the stopping time $t$ relies on the oracle information of the true parameter $\mathbf{w}^*$. Therefore the "early-stopped GD" in Theorem 3.1 is not a practical algorithm. Instead, we should view Theorem 3.1 as a guarantee for GD with an optimally tuned stopping time. It will also be clear later in Section 4 that the optimal stopping time $t$ must be finite for overparameterized logistic regression. Moreover, we point out that the stopping time $t$ is a function of $k$ and thus also depends on the sample size $n$.

Although the stopping time in Theorem 3.1 is implicit, one can compute an upper bound on it using standard optimization and concentration tools. Specifically, GD converges in $\mathcal{O}(1/t)$ rate as the empirical risk is convex and smooth. Moreover, we can compute $\widehat{\mathcal{L}}(\mathbf{w}_{0:k})$ using concentration bounds. These lead to an upper bound on the stopping time.

**Misspecification.** For the simplicity of discussion, we state Theorem 3.1 in a well-specified case formalized by Assumption 1. Nonetheless, from its proof in Appendix B.1, it is clear that the same results also hold in misspecified cases, where we define $\mathbf{w}^* \in \arg\min \mathcal{L}$ and assume $\boldsymbol{\Sigma}^{-1/2}\mathbf{x}$ is subGaussian. In those misspecified cases, however, Proposition 2.1 may not hold. Thus Theorem 3.1 only provides a logistic risk bound but does not yield any bounds on calibration error or zero-one error.

We also note that the proof of Theorem 3.1 can be adapted to other loss functions that are convex, smooth, and Lipschitz.

### 3.2. A Variance-Dominating Bound

From Theorem 3.1, we see that early-stopped GD is consistent and calibrated under the arguably weakest condition on the true parameter, $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} < \infty$. However, the attained bound decays at a rate no faster than $\mathcal{O}(1/\sqrt{n})$ as long as $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} \gtrsim 1$. In the simpler case where $\|\mathbf{w}^*\| < \infty$, we can tune the stopping time to achieve an improved bound.

> **Theorem 3.2 (A "variance-dominating" risk bound).** *Suppose that Assumption 1 holds with $\|\mathbf{w}^*\| < \infty$. Let $k$ be an arbitrary index. Suppose that the stepsize for GD satisfies the same condition as in Theorem 3.1 and the stopping time $t$ is such that*
> $$\widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{w}^*) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1}).$$
> *Assume for simplicity that $\|\mathbf{w}^*\| \gtrsim 1$, $\lambda_1 \lesssim 1$, and $\mathrm{tr}(\boldsymbol{\Sigma}) \gtrsim 1$. Then with probability at least $1 - \delta$, we have*
> $$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} \lesssim \|\mathbf{w}^*\| \left(\frac{k}{n} + \sqrt{\frac{\sum_{i>k} \lambda_i}{n}} + \frac{\mathrm{tr}(\boldsymbol{\Sigma})^{1/2} \ln\!\left(n\|\mathbf{w}^*\|\,\mathrm{tr}(\boldsymbol{\Sigma})/\delta\right)}{n}\right).$$

**Comparing Theorems 3.1 and 3.2.** Compared to Theorem 3.1, Theorem 3.2 achieves a faster rate, but is only applicable when $\|\mathbf{w}^*\| < \infty$. Specifically, in the classical finite-dimensional setting where $\|\mathbf{w}^*\| \asymp 1$ and $\boldsymbol{\Sigma} = \mathbf{I}_d$, the excess risk bound in Theorem 3.2 decreases at the rate of $\tilde{\mathcal{O}}(d/n)$ while that in Theorem 3.1 decreases at the rate of $\tilde{\mathcal{O}}(\sqrt{d/n})$. For another example, under the source and capacity conditions of (1), Theorem 3.2 provides an improved excess risk bound of $\tilde{\mathcal{O}}(n^{-a/(1+a)})$ when $b > a+1$, but is not applicable when $b \leq a+1$.

The stopping time in Theorem 3.1 is designed to handle more general high-dimensional situations that even allow $\|\mathbf{w}^*\| = \infty$. It tends to stop "earlier" so that the bias error tends to dominate the variance error. In comparison, Theorem 3.2 is limited to simpler cases where $\|\mathbf{w}^*\| < \infty$ and sets a "later" stopping time so that the variance error tends to dominate the bias error. Therefore Theorem 3.2 achieves a faster rate.

**Future directions.** Theorems 3.1 and 3.2 are sufficiently powerful for our purpose of demonstrating the benefits of early stopping. However, we point out that neither Theorems 3.1 nor 3.2 reveal the true trade-off between the bias and variance errors induced by early stopping. This is unsatisfactory given that in linear regression, the exact trade-off between bias and variance errors has been settled for one-pass SGD (Zou et al., 2023; Wu et al., 2022a;b) and $\ell_2$-regularization (Tsigler & Bartlett, 2023), and has been partially settled for early-stopped GD (Zou et al., 2022, assuming a Gaussian prior). We leave the improvement of these bounds for future work.

From a technical perspective, the gap in analysis between linear regression and logistic regression is significant. All the prior sharp analyses of GD in linear regression make heavy use of explicit calculations with chains of equalities and closed-form solutions. But these fail to hold for GD in logistic regression since the Hessian is no longer fixed. While one might suspect that a limiting analogy can be made where least squares ideas are applied locally around an optimum, a priori there is no reason to believe that the GD path, which diverges to infinity, even passes near the population optimum, let alone spends a reasonable amount of time there. Moreover, as our lower bounds in Section 4 attest, the GD path exhibits significant curvature. Due to these issues, we believe tools from linear regression can not be merely ported over, and new approaches are required. While we have provided some tools to this end, Theorems 3.1 and 3.2 do not tightly characterize the GD path, and much is left to future work.

### 3.3. Implicit Bias of Early Stopping

In this part, we briefly discuss the proof ideas by introducing the following key lemma in our analysis.

> **Lemma 3.3 (Implicit bias of early stopping).** *Let $\widehat{\mathcal{L}}(\cdot)$ be convex and $\beta$-smooth. Let $(\mathbf{w}_t)_{t \geq 0}$ be given by* (GD) *with stepsize $\eta \leq 1/\beta$. Then for every $\mathbf{u}$, we have*
> $$\frac{\|\mathbf{w}_t - \mathbf{u}\|^2}{2\eta t} + \widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{u}) + \frac{\|\mathbf{u}\|^2}{2\eta t}, \quad t > 0.$$

This lemma reveals an implicit bias of early-stopping, in which early-stopped GD *attains a small empirical risk while maintaining a relatively small norm*. Specifically, consider a comparator $\mathbf{u}$ and a stopping time $t$ such that

$$\widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{u}) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1}).$$

This stopping time together with Lemma 3.3 (applied to $t-1$) leads to

$$\widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{u}), \quad \text{and} \quad \|\mathbf{w}_{t-1} - \mathbf{u}\| \leq \|\mathbf{u}\|.$$

By optimizing the choice of the comparator $\mathbf{u}$, we see that early-stopped GD achieves a small empirical risk with a relatively small norm.

Besides Lemma 3.3, the remaining efforts for proving Theorems 3.1 and 3.2 are using classical tools of Rademacher complexity (Bartlett & Mendelson, 2002; Kakade et al., 2008) and local Rademacher complexity (Bartlett et al., 2005), respectively.

Later in Section 5, we will use Lemma 3.3 to show connections between early stopping and $\ell_2$-regularization.

---

## 4. Lower Bounds for Interpolating Estimators

In this section, we provide negative results for interpolating estimators by establishing risk lower bounds for them.

### 4.1. Logistic Risk and Calibration Error

The following theorem shows that GD without early stopping must induce an unbounded logistic risk and a positive calibration error in the overparameterized regime.

> **Theorem 4.1 (Lower bounds for logistic risk and calibration error).** *Suppose that Assumption 1 holds. Let $\tilde{\mathbf{w}}$ be a unit vector such that $\|\tilde{\mathbf{w}}\|_{\boldsymbol{\Sigma}} > 0$ and let $(\mathbf{w}_t)_{t \geq 0}$ be a sequence of vectors such that*
> $$\|\mathbf{w}_t\| \to \infty, \quad \frac{\mathbf{w}_t}{\|\mathbf{w}_t\|} \to \tilde{\mathbf{w}}.$$
> *Then we have*
> $$\lim_{t \to \infty} \mathcal{C}(\mathbf{w}_t) \geq \exp(-C\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}), \quad \lim_{t \to \infty} \mathcal{L}(\mathbf{w}_t) = \infty,$$
> *where $C > 1$ is a constant.*

Theorem 4.1 shows that for every sequence of estimators that diverges in norm but converges in direction, their induced logistic risk must grow unboundedly and their induced calibration error must be bounded away from zero by a constant. Therefore, their limit is *inconsistent* (for logistic risk) and *poorly calibrated*. According to Proposition 2.2, this applies to GD iterates in the overparameterized regime.

Combining this with our preceding discussion, we see that for every well-specified but overparameterized logistic regression problem, GD is calibrated and consistent (for logistic risk) when early stopped, but is poorly calibrated and inconsistent (for logistic risk) at convergence. This contrast demonstrates the benefit of early stopping.

### 4.2. Zero-One Error

The preceding lower bounds in Theorem 4.1 are tied to the divergence of the norm of the estimators. In this part, we show that even when properly normalized, interpolating estimators are still inferior to early-stopped GD. To this end, we consider the zero-one error that is insensitive to the estimator norm.

> **Theorem 4.2 (A lower bound for zero-one error).** *Suppose that Assumption 1 holds. Let $C_2 > C_1 > 1$ be two sufficiently large constants. Assume that $\boldsymbol{\Sigma}^{1/2}\mathbf{w}^*$ is $k$-sparse and $1/C_1 \leq \|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} \leq C_1$. Assume that*
> $$n \geq C_1 k \ln(k/\delta), \quad C_1 \leq \frac{\mathrm{rank}(\boldsymbol{\Sigma})}{n \ln(n) \ln(1/\delta)} \leq C_2.$$
> *Then with probability at least $1 - \delta$, for every interpolating estimator $\hat{\mathbf{w}}$ such that $\min_{i \in [n]} y_i \mathbf{x}_i^\top \hat{\mathbf{w}} > 0$, we have*
> $$\mathcal{E}(\hat{\mathbf{w}}) - \min\mathcal{E} \gtrsim \frac{1}{\sqrt{\ln(n)\ln(1/\delta)}}.$$

Theorem 4.2 characterizes a class of overparameterized logistic regression problems where every interpolating estimator needs at least an *exponential* number of training data to achieve a small excess zero-one error. This applies to asymptotic GD as it converges to the maximum $\ell_2$-margin solution by Proposition 2.2. In contrast, Theorems 3.1 and 3.2 suggest that early-stopped GD can achieve a small excess zero-one error using at most a *polynomial* number of training data under weak conditions. These weak conditions can be, for example, $\|\mathbf{w}^*\| < \infty$ or the sparsity parameter $k$ does not grow with $n$ (see also the examples given by (1)). This separation underscores the benefits of early stopping for reducing sample complexity.

The intuition behind Theorem 4.2 is that there are $k$ informative dimensions and a lot more uninformative dimensions. Since $n \gg k$, the training set cannot be separated purely using the $k$ informative dimensions. Thus, interpolators must use the uninformative dimensions to separate the data, leading to the risk lower bound.

**Future direction.** Note that Theorem 4.2 applies to every interpolating estimator. When restricted to the maximum $\ell_2$-margin estimator, the one that GD converges to in direction, we conjecture that a *constant* lower bound on the excess zero-one error can be proved, especially when the spectrum of the data covariance matrix decays fast. This is left for future investigation.

---

## 5. Early Stopping and $\ell_2$-Regularization

Sections 3 and 4 demonstrate that early stopping carries a certain regularization effect that benefits its statistical performance. This regularization is, however, implicit. In this section, we attempt to provide some intuitions of the implicit regularization of early stopping by establishing its connections to an explicit, $\ell_2$-regularization. An $\ell_2$-regularized *empirical risk minimizer* (ERM) is defined as

$$\mathbf{u}_\lambda := \arg\min_{\mathbf{u}} \widehat{\mathcal{L}}(\mathbf{u}) + \frac{\lambda}{2}\|\mathbf{u}\|^2, \tag{2}$$

where $\lambda > 0$ is the regularization strength. Note that $\mathbf{u}_\lambda$ is unique and well-defined as long as $\widehat{\mathcal{L}}(\cdot)$ is convex, whereas $\widehat{\mathcal{L}}(\cdot)$ does not have to have a finite minimizer. We refer to $(\mathbf{u}_\lambda)_{\lambda > 0}$ given by (2) as the *$\ell_2$-regularization path*. Similarly, we refer to $(\mathbf{w}_t)_{t > 0}$ given by (GD) as the *GD path*.

In linear regression, prior works showed that the excess risk of early-stopped GD (and one-pass SGD) is comparable to that of $\ell_2$-regularized ERM (Ali et al., 2019; Zou et al., 2021). For strongly convex and smooth problems, Suggala et al. (2018) provided bounds on the $\ell_2$-distance between the GD and $\ell_2$-regularization paths. In what follows, we establish more connections between the GD and $\ell_2$-regularization paths. We first establish a relative but global connection in convex (not necessarily strongly convex) and smooth problems, then we establish an asymptotic but absolute connection in overparameterized logistic regression problems.

### 5.1. A Global Connection

The following theorem presents a global comparison of the norm and angle between the GD and $\ell_2$-regularization paths.

> **Theorem 5.1 (A global bound).** *Let $\widehat{\mathcal{L}}(\cdot)$ be convex and $\beta$-smooth. Consider $(\mathbf{w}_t)_{t \geq 0}$ given by* (GD) *with stepsize $\eta \leq 1/\beta$ and $(\mathbf{u}_\lambda)_{\lambda > 0}$ given by (2). Set $\lambda := 1/(\eta t)$. Then we have*
> $$\text{for every } t > 0, \quad \|\mathbf{w}_t - \mathbf{u}_\lambda\| \leq \frac{1}{\sqrt{2}}\|\mathbf{w}_t\|.$$
> *As a direct consequence, the following holds for every $t > 0$:*
> $$\cos(\mathbf{w}_t, \mathbf{u}_\lambda) \geq \frac{1}{\sqrt{2}}, \quad \frac{\sqrt{2}}{1+\sqrt{2}}\|\mathbf{u}_\lambda\| \leq \|\mathbf{w}_t\| \leq \frac{\sqrt{2}}{\sqrt{2}-1}\|\mathbf{u}_\lambda\|.$$

Theorem 5.1 establishes a global but relative connection between the GD and $\ell_2$-regularization paths for all convex and smooth problems. Specifically, starting from the same zero initialization, the angle between the two paths is no more than $\pi/4$, and the norm of the two paths differs by a factor within 0.585 and 3.415. We point out this relative connection holds *globally* for every stopping time (with its corresponding regularization strength) and for every convex and smooth problem. In particular, it applies to overparameterized logistic regression, which is smooth and convex but not strongly convex. We also note that using the norm bounds in Theorem 5.1, the upper bounds in Theorems 3.1 and 3.2 for early-stopped GD can be easily adapted to $\ell_2$-regularized ERM.

Theorem 5.1 cannot be improved without making further assumptions. This is because the GD and $\ell_2$-regularization paths could converge to two distinct limits (as $t \to \infty$ and $\lambda \to 0$) in convex but non-strongly convex problems (see Suggala et al., 2018, Section 4). So in general, we cannot expect their distance to be small in the absolute sense.

### 5.2. An Asymptotic Comparison

We have established a global but relative connection between the GD and $\ell_2$-regularization paths in Theorem 5.1. We now turn to logistic regression with linearly separable data and establish an absolute but asymptotic connection between the two paths.

In logistic regression with linearly separable data, both GD and $\ell_2$-regularization paths diverge to infinity in norm (as $t \to \infty$ and $\lambda \to 0$) while converging in direction to the maximum $\ell_2$-margin solution (Rosset et al., 2004; Soudry et al., 2018; Ji & Telgarsky, 2018; Ji et al., 2020). Therefore their angle tends to zero asymptotically (Suggala et al., 2018; Ji et al., 2020). This characterization is more precise than the $\pi/4$ global angle bound from Theorem 5.1.

However, it remains unclear how the $\ell_2$-distance between the two paths evolves in logistic regression with linearly separable data. Quite surprisingly, we will show that their $\ell_2$-distance tends to zero under a widely used condition (Soudry et al., 2018; Ji & Telgarsky, 2021; Wu et al., 2023), but could diverge to infinity in the worst case.

Let $\mathbf{X} := (\mathbf{x}_1, \ldots, \mathbf{x}_n)^\top$ and $\mathbf{y} := (y_1, \ldots, y_n)^\top$ be a set of linearly separable data. Then the Lagrangian dual of the margin maximization program in Proposition 2.2 is given by (see Hsu et al., 2021, for example)

$$\max_{\boldsymbol{\beta} \in \mathbb{R}^n} -\frac{1}{2}\boldsymbol{\beta}^\top \mathbf{X}\mathbf{X}^\top \boldsymbol{\beta} + \boldsymbol{\beta}^\top \mathbf{y} \quad \text{s.t. } y_i \beta_i \geq 0, \; i \in [n].$$

Here, $\boldsymbol{\beta}$ are the dual variables multiplied by $\mathbf{y}$ entry-wise. Let $\hat{\boldsymbol{\beta}}$ be the solution to the above problem. Let $S_+ := \{i \in [n] : y_i \hat{\beta}_i > 0\}$ be the set of support vectors (with strictly positive dual variables). The following condition assumes the coverage of the support vectors.

> **Assumption 2 (Support vectors condition).** *Assume that $\mathrm{rank}\{\mathbf{x}_i : i \in S_+\} = \mathrm{rank}\{\mathbf{x}_i : i \in [n]\}$.*

Assumption 2 has been widely used in the analysis of the implicit bias (Soudry et al., 2018; Ji & Telgarsky, 2021; Wu et al., 2023). In particular, Assumption 2 holds if every data is a support vector, which is common in high-dimensional situations (Hsu et al., 2021; Wang & Thrampoulidis, 2022; Cao et al., 2021).

> **Theorem 5.2 (An asymptotic bound).** *Let $(\mathbf{x}_i, y_i)_{i=1}^n$ be a linearly separable dataset that satisfies Assumption 2. Let $(\mathbf{w}_t)_{t>0}$ and $(\mathbf{u}_\lambda)_{\lambda>0}$ be the GD and $\ell_2$-regularization paths, respectively, for logistic regression with $(\mathbf{x}_i, y_i)_{i=1}^n$. Then there exists $\lambda$ as a function of $t$ such that*
> $$\|\mathbf{w}_t - \mathbf{u}_{\lambda(t)}\| \to 0, \quad \text{while } \|\mathbf{w}_t\|, \|\mathbf{u}_{\lambda(t)}\| \to \infty, \quad \text{as } t \to \infty.$$

For logistic regression with linearly separable data under Assumption 2, Theorem 5.2 shows that the $\ell_2$-distance between the GD and $\ell_2$-regularization paths tends to zero, despite that both paths diverge to infinity in their norm. Note that this implies their angle converges to zero, and is more precise than the relative norm bound from Theorem 5.1.

However, this sharp asymptotic connection is strongly tied to Assumption 2. Surprisingly, when Assumption 2 fails to hold, the $\ell_2$-distance between the GD and $\ell_2$-regularization paths could tend to infinity instead.

> **Theorem 5.3 (A counter example).** *Consider the following dataset*
> $$\mathbf{x}_1 := (\gamma, 0)^\top,\; y_1 := 1, \quad \mathbf{x}_2 := (\gamma, \gamma_2)^\top,\; y_2 := 1,$$
> *where $0 < \gamma_2 < \gamma < 1$. Then $(\mathbf{x}_i, y_i)_{i=1,2}$ is linearly separable but violates Assumption 2. Let $(\mathbf{w}_t)_{t \geq 0}$ and $(\mathbf{u}_\lambda)_{\lambda \geq 0}$ be the GD and $\ell_2$-regularization paths respectively for logistic regression with $(\mathbf{x}_i, y_i)_{i=1,2}$. Then $\|\mathbf{w}_t\| \to \infty$ as $t \to \infty$. Moreover, for every map $\lambda : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$, we have*
> $$\|\mathbf{w}_t - \mathbf{u}_{\lambda(t)}\| \gtrsim \ln\ln(\|\mathbf{w}_t\|) \to \infty.$$

This simple yet strong counter-example suggests that the $\ell_2$-distance between the GD and $\ell_2$-regularization path can diverge to infinity when Assumption 2 fails to hold.

**Future directions.** We conjecture that for logistic regression with linearly separable data, the limit of the $\ell_2$-distance between the GD and $\ell_2$-regularized paths is either zero or infinity, and the phase transition is determined by a certain geometric property of the dataset (for example, Assumption 2). The reasoning behind this conjecture is as follows. Note that Assumption 2 implies that the dataset projected perpendicular to the max-margin direction (called "projected dataset") is strictly nonseparable (Wu et al., 2023, Lemma 3.1). This is the main property used in Theorem 5.2. Moreover, in Theorem 5.3, the "projected dataset" is nonseparable but with margin zero—we conjecture this property is sufficient for Theorem 5.3 to hold. Now for a generic separable dataset, we check the "projected dataset": if it is strictly nonseparable, Theorem 5.2 holds; if it is nonseparable but with margin zero, we conjecture Theorem 5.3 holds; otherwise, it is separable (with positive margin), we decompose the dataset recursively. This is the reasoning behind our conjecture.

It also remains unclear to what extent early stopping replicates the effects of explicit regularization for logistic regression. Specifically, is there a logistic regression example such that early-stopped GD has a better calibration/logistic risk rate than $\ell_2$-regularization or vice-versa? This is left for future investigation, as our current bounds are not sharp enough to yield a concrete answer.

---

## 6. Related Works

**Benign overfitting in logistic regression.** A line of work shows the benign overfitting of the asymptotic GD (or the maximum $\ell_2$-margin estimator) in overparameterized logistic regression under a variety of assumptions (Montanari et al., 2019; Chatterji & Long, 2021; Cao et al., 2021; Wang & Thrampoulidis, 2022; Muthukumar et al., 2021; Shamir, 2023). Our results are not a violation of theirs; instead, we show an additional regularization of early-stopping, which brings statistical advantages of early-stopped GD over asymptotic GD such as calibration and a smaller sample complexity.

**M-estimators for logistic regression.** In the classical finite $d$-dimensional setting, the sample complexity of the empirical risk minimizer (ERM) for logistic regression is well-studied (Ostrovskii & Bach, 2021; Kuchelmeister & van de Geer, 2024; Hsu & Mazumdar, 2024; Chardon et al., 2024), where the minimax rate is known to be $\mathcal{O}(d/n)$. Different from theirs, we focus on an overparameterized regime, where the ERM of logistic regression does not even exist. When specialized to their setting, our Theorem 3.2 recovers the comparable $\tilde{\mathcal{O}}(d/n)$ rate.

In the nonparametric setting, the works by (Bach, 2010; Marteau-Ferey et al., 2019) provided logistic risk bounds for $\ell_2$-regularized ERM. Bach (2010) only considered a fixed design setting, whereas Marteau-Ferey et al. (2019) required that $\|\mathbf{w}^*\| < \infty$. Different from theirs, we aim to understand the benefits of the implicit regularization of early-stopping, instead of that of explicit $\ell_2$-regularization. Moreover, we show that early-stopped GD achieves a vanishing excess logistic risk as long as $\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} < \infty$, without assuming a finite $\|\mathbf{w}^*\|$. In the regimes where our results are directly comparable, however, our risk bounds might be less tight than theirs.

The work by Bach (2014) considered one-pass SGD for logistic regression assuming strong convexity around the true model parameter. This strong convexity assumption, however, is prohibitive in our high-dimensional settings.

There is a line of works (Sur & Candès, 2019; Candès & Sur, 2020) focused on the existence of ERM for logistic regression in a proportional limit setting (assuming that $n, d \to \infty$ in a fixed ratio). This is quite apart from our focus, where ERM never exists due to overparameterization.

**Separable distribution.** There are logistic risk bounds of early-stopped GD (and one-pass SGD) developed in the *noiseless* cases, assuming a separable population distribution (Ji & Telgarsky, 2018; Shamir, 2021; Telgarsky, 2022; Schliserman & Koren, 2024). These results do not imply any benefits of early stopping, as their setting is noiseless. In comparison, we consider overparameterized logistic regression with a strictly non-separable population distribution, where the risk of overfitting is prominent. In this case, our results suggest that early stopping plays a significant role in preventing overfitting.

**Early stopping for classification.** In the boosting literature, an early work by Zhang & Yu (2005) showed that boosting methods (that can be interpreted as coordinate descent) with early stopping are consistent in the classification sense; related refined studies for boosting with the squared loss with early stopping were also provided by Bühlmann & Yu (2003). The paper is also notable for giving the first proof of boosting methods converging to the maximum margin solution (Zhang & Yu, 2005, Appendix D), which was later refined with rates by (Telgarsky, 2013). Their results can be converted to GD. In particular, related concepts were used to prove consistency of early-stopped GD for shallow networks in the lazy regime (Ji et al., 2021). In contrast with the present work that focuses on high-dimensional cases, the preceding works only deal with finite-dimensional settings. Moreover, none of those works provide lower bounds for interpolating estimators and tight links to the regularization path which are provided in the present work.

**Classification calibration.** Proposition 2.1 captures a very nice consequence of logistic loss minimization: *calibration* and *classification-calibration*, respectively recovery of the optimal conditional probability model and of the optimal classifier. For more general convex losses, the ability to construct a general conditional probability model was developed by Zhang (2004) as a conceptual tool in establishing classification calibration, but without explicitly controlling calibration error. A further abstract treatment of classification calibration was later presented by Bartlett et al. (2006). The refined statistical rates, separations, and early-stopping consequences studied in the present work were not considered in those works.

---

## 7. Conclusion

We show the benefits of early stopping in GD for overparameterized and well-specified logistic regression. We show that for every well-specified logistic regression problem, early-stopped GD is calibrated while asymptotic GD is not. Furthermore, we show that early-stopped GD achieves a small excess zero-one error with only a polynomial number of samples, in contrast to interpolating estimators, including asymptotic GD, which require an exponential number of samples to achieve the same. Finally, we establish nonasymptotic bounds on the differences between the GD and the $\ell_2$-regularization paths. Altogether, we underscore the statistical benefits of early stopping, partially explained by its connection with $\ell_2$-regularization.

---

## Acknowledgments

We gratefully acknowledge the support of the NSF for FODSI through grant DMS-2023505, of the NSF and the Simons Foundation for the Collaboration on the Theoretical Foundations of Deep Learning through awards DMS-2031883 and #814639, of the NSF through grants DMS-2209975 and DMS-2413265, and of the ONR through MURI award N000142112431. The authors are also grateful to the Simons Institute for hosting them during parts of this work.

---

## References

- Ali, A., Kolter, J. Z., and Tibshirani, R. J. A continuous-time view of early stopping for least squares regression. In *The 22nd International Conference on Artificial Intelligence and Statistics*, pp. 1370–1378. PMLR, 2019.
- Bach, F. Self-concordant analysis for logistic regression. *Electronic Journal of Statistics*, 4:384–414, 2010.
- Bach, F. Adaptivity of averaged stochastic gradient descent to local strong convexity for logistic regression. *The Journal of Machine Learning Research*, 15(1):595–627, 2014.
- Bartlett, P. and Shawe-Taylor, J. Generalization performance of support vector machines and other pattern classifiers. *Advances in Kernel methods—support vector learning*, pp. 43–54, 1999.
- Bartlett, P. L. and Mendelson, S. Rademacher and Gaussian complexities: Risk bounds and structural results. *Journal of Machine Learning Research*, 3(Nov):463–482, 2002.
- Bartlett, P. L., Bousquet, O., and Mendelson, S. Local Rademacher complexities. *Annals of Statistics*, pp. 1497–1537, 2005.
- Bartlett, P. L., Jordan, M. I., and McAuliffe, J. D. Convexity, classification, and risk bounds. *Journal of the American Statistical Association*, 101(473):138–156, 2006.
- Bartlett, P. L., Long, P. M., Lugosi, G., and Tsigler, A. Benign overfitting in linear regression. *Proceedings of the National Academy of Sciences*, 117(48):30063–30070, 2020.
- Bartlett, P. L., Montanari, A., and Rakhlin, A. Deep learning: a statistical viewpoint. *Acta Numerica*, 30:87–201, 2021.
- Bühlmann, P. and Yu, B. Boosting with the $\ell_2$ loss: regression and classification. *Journal of the American Statistical Association*, 98(462):324–339, 2003.
- Candès, E. J. and Sur, P. The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression. *The Annals of Statistics*, 48(1):27–42, 2020.
- Cao, Y., Gu, Q., and Belkin, M. Risk bounds for overparameterized maximum margin classification on subgaussian mixtures. *Advances in Neural Information Processing Systems*, 34:8407–8418, 2021.
- Caponnetto, A. and De Vito, E. Optimal rates for the regularized least-squares algorithm. *Foundations of Computational Mathematics*, 7:331–368, 2007.
- Chardon, H., Lerasle, M., and Mourtada, J. Finite-sample performance of the maximum likelihood estimator in logistic regression. arXiv preprint arXiv:2411.02137, 2024.
- Chatterji, N. S. and Long, P. M. Finite-sample analysis of interpolating linear classifiers in the overparameterized regime. *Journal of Machine Learning Research*, 22(129):1–30, 2021.
- Dieuleveut, A. and Bach, F. Nonparametric stochastic approximation with large step-sizes. *The Annals of Statistics*, 44(4):1363–1399, 2016.
- Foster, D. P. and Vohra, R. V. Asymptotic calibration. *Biometrika*, 85(2):379–390, 1998.
- Hsu, D. and Mazumdar, A. On the sample complexity of parameter estimation in logistic regression with normal design. In *Proceedings of Thirty Seventh Conference on Learning Theory*, volume 247 of PMLR, pp. 2418–2437, 2024.
- Hsu, D., Muthukumar, V., and Xu, J. On the proliferation of support vectors in high dimensions. In *International Conference on Artificial Intelligence and Statistics*, pp. 91–99. PMLR, 2021.
- Ji, Z. and Telgarsky, M. Risk and parameter convergence of logistic regression. arXiv preprint arXiv:1803.07300, 2018.
- Ji, Z. and Telgarsky, M. Polylogarithmic width suffices for gradient descent to achieve arbitrarily small test error with shallow ReLU networks. In *International Conference on Learning Representations*, 2019.
- Ji, Z. and Telgarsky, M. Characterizing the implicit bias via a primal-dual analysis. In *Algorithmic Learning Theory*, pp. 772–804. PMLR, 2021.
- Ji, Z., Dudík, M., Schapire, R. E., and Telgarsky, M. Gradient descent follows the regularization path for general losses. In *Conference on Learning Theory*, pp. 2109–2136. PMLR, 2020.
- Ji, Z., Li, J., and Telgarsky, M. Early-stopped neural networks are consistent. *Advances in Neural Information Processing Systems*, 34:1805–1817, 2021.
- Kakade, S. M., Sridharan, K., and Tewari, A. On the complexity of linear prediction: Risk bounds, margin bounds, and regularization. *Advances in Neural Information Processing Systems*, 21, 2008.
- Kuchelmeister, F. and van de Geer, S. Finite sample rates for logistic regression with small noise or few samples. *Sankhya A*, pp. 1–70, 2024.
- Lin, J. and Rosasco, L. Optimal rates for multi-pass stochastic gradient methods. *Journal of Machine Learning Research*, 18(97):1–47, 2017.
- Marteau-Ferey, U., Ostrovskii, D., Bach, F., and Rudi, A. Beyond least-squares: Fast rates for regularized empirical risk minimization through self-concordance. In *Conference on Learning Theory*, pp. 2294–2340. PMLR, 2019.
- Mohri, M., Rostamizadeh, A., and Talwalkar, A. *Foundations of Machine Learning*. MIT Press, 2018.
- Montanari, A., Ruan, F., Sohn, Y., and Yan, J. The generalization error of max-margin linear classifiers: Benign overfitting and high dimensional asymptotics in the overparameterized regime. arXiv preprint arXiv:1911.01544, 2019.
- Muthukumar, V., Narang, A., Subramanian, V., Belkin, M., Hsu, D., and Sahai, A. Classification vs regression in overparameterized regimes: Does the loss function matter? *Journal of Machine Learning Research*, 22(222):1–69, 2021.
- Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N. Exploring generalization in deep learning. *Advances in Neural Information Processing Systems*, 30, 2017.
- Ostrovskii, D. M. and Bach, F. Finite-sample analysis of M-estimators using self-concordance. *Electronic Journal of Statistics*, 15:326–391, 2021.
- Rosset, S., Zhu, J., and Hastie, T. Boosting as a regularized path to a maximum margin classifier. *The Journal of Machine Learning Research*, 5:941–973, 2004.
- Schliserman, M. and Koren, T. Tight risk bounds for gradient descent on separable data. *Advances in Neural Information Processing Systems*, 36, 2024.
- Shamir, O. Gradient methods never overfit on separable data. *Journal of Machine Learning Research*, 22(85):1–20, 2021.
- Shamir, O. The implicit bias of benign overfitting. *Journal of Machine Learning Research*, 24(113):1–40, 2023.
- Sonthalia, R., Lok, J., and Rebrova, E. On regularization via early stopping for least squares regression. arXiv preprint arXiv:2406.04425, 2024.
- Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N. The implicit bias of gradient descent on separable data. *The Journal of Machine Learning Research*, 19(1):2822–2878, 2018.
- Suggala, A., Prasad, A., and Ravikumar, P. K. Connecting optimization and regularization paths. *Advances in Neural Information Processing Systems*, 31, 2018.
- Sur, P. and Candès, E. J. A modern maximum-likelihood theory for high-dimensional logistic regression. *Proceedings of the National Academy of Sciences*, 116(29):14516–14525, 2019.
- Telgarsky, M. Margins, shrinkage, and boosting. In *International Conference on Machine Learning*, pp. 307–315. PMLR, 2013.
- Telgarsky, M. Stochastic linear optimization never overfits with quadratically-bounded losses on general data. In *Conference on Learning Theory*, pp. 5453–5488. PMLR, 2022.
- Tsigler, A. and Bartlett, P. L. Benign overfitting in ridge regression. *Journal of Machine Learning Research*, 24(123):1–76, 2023.
- Wang, K. and Thrampoulidis, C. Binary classification of Gaussian mixtures: Abundance of support vectors, benign overfitting, and regularization. *SIAM Journal on Mathematics of Data Science*, 4(1):260–284, 2022.
- Wu, J., Zou, D., Braverman, V., Gu, Q., and Kakade, S. Last iterate risk bounds of SGD with decaying stepsize for overparameterized linear regression. In *International Conference on Machine Learning*, pp. 24280–24314. PMLR, 2022a.
- Wu, J., Zou, D., Braverman, V., Gu, Q., and Kakade, S. M. The power and limitation of pretraining-finetuning for linear regression under covariate shift. *The 36th Conference on Neural Information Processing Systems*, 2022b.
- Wu, J., Braverman, V., and Lee, J. D. Implicit bias of gradient descent for logistic regression at the edge of stability. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
- Wu, J., Bartlett, P. L., Telgarsky, M., and Yu, B. Large stepsize gradient descent for logistic loss: Non-monotonicity of the loss improves optimization efficiency. *Conference on Learning Theory*, 2024.
- Yao, Y., Rosasco, L., and Caponnetto, A. On early stopping in gradient descent learning. *Constructive Approximation*, 26(2):289–315, 2007.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3):107–115, 2021.
- Zhang, T. Statistical behavior and consistency of classification methods based on convex risk minimization. *The Annals of Statistics*, 32(1):56–85, 2004.
- Zhang, T. and Yu, B. Boosting with early stopping: Convergence and consistency. *Annals of Statistics*, pp. 1538–1579, 2005.
- Zou, D., Wu, J., Braverman, V., Gu, Q., Foster, D. P., and Kakade, S. The benefits of implicit regularization from SGD in least squares problems. *Advances in Neural Information Processing Systems*, 34:5456–5468, 2021.
- Zou, D., Wu, J., Braverman, V., Gu, Q., and Kakade, S. Risk bounds of multi-pass SGD for least squares in the interpolation regime. *Advances in Neural Information Processing Systems*, 35:12909–12920, 2022.
- Zou, D., Wu, J., Braverman, V., Gu, Q., and Kakade, S. M. Benign overfitting of constant-stepsize SGD for linear regression. *Journal of Machine Learning Research*, 24(326):1–58, 2023.

---

## Appendix A. Basic Results

### A.1. Proof of Proposition 2.1

*Proof.* We first compute the logistic loss. Define

$$p_\mathbf{x}(\mathbf{w}) := \frac{1}{1 + \exp(-\mathbf{x}^\top\mathbf{w})}, \quad p^*_\mathbf{x} := \frac{1}{1 + \exp(-\mathbf{x}^\top\mathbf{w}^*)}.$$

Then under Assumption 1 we have

$$\mathcal{L}(\mathbf{w}) = \mathbb{E}_{\mathbf{x},y}\ln(1 + \exp(-y\mathbf{x}^\top\mathbf{w}))
= -\mathbb{E}_\mathbf{x}\!\left(p^*_\mathbf{x}\ln p_\mathbf{x}(\mathbf{w}) + (1-p^*_\mathbf{x})\ln(1-p_\mathbf{x}(\mathbf{w}))\right)
= \mathbb{E}_\mathbf{x}\!\left(H(p^*_\mathbf{x}) + \mathrm{KL}(p^*_\mathbf{x} \| p_\mathbf{x}(\mathbf{w}))\right),$$

where the first term is the entropy of a Bernoulli distribution with head probability $p^*_\mathbf{x}$, and the second term is the KL divergence between two Bernoulli distributions with head probabilities $p^*_\mathbf{x}$ and $p_\mathbf{x}(\mathbf{w})$, respectively. It is then clear that $\mathbf{w}^*$ is the unique minimizer of $\mathcal{L}(\mathbf{w})$.

We then compute the zero-one error under Assumption 1. Note that $\mathcal{E}(\mathbf{0}) = 1$. If $\mathbf{w}^* = \mathbf{0}$, then for every $\mathbf{w}$,

$$\mathcal{E}(\mathbf{w}) = \mathbb{E}_{\mathbf{x},y}\mathbf{1}[y\mathbf{x}^\top\mathbf{w} \leq 0] = \mathbb{E}_\mathbf{x}\, 0.5\!\left(\mathbf{1}[\mathbf{x}^\top\mathbf{w} \leq 0] + \mathbf{1}[\mathbf{x}^\top\mathbf{w} \geq 0]\right) \geq 1 = \mathcal{E}(\mathbf{w}^*),$$

so $\mathbf{w}^* \in \arg\min \mathcal{E}(\cdot)$. If $\mathbf{w}^* \neq \mathbf{0}$, we have $\mathbf{x}^\top\mathbf{w}^* = 0$ is measure zero. Then

$$\mathcal{E}(\mathbf{w}^*) = \mathbb{E}_\mathbf{x}\!\left(p^*_\mathbf{x}\mathbf{1}[\mathbf{x}^\top\mathbf{w}^* \leq 0] + (1-p^*_\mathbf{x})\mathbf{1}[\mathbf{x}^\top\mathbf{w}^* \geq 0]\right) = \mathbb{E}_\mathbf{x}\min\{p^*_\mathbf{x}, 1-p^*_\mathbf{x}\} \leq 0.5 < \mathcal{E}(\mathbf{0}).$$

It remains to check that $\mathcal{E}(\mathbf{w}^*) \leq \mathcal{E}(\mathbf{w})$ for all $\mathbf{w} \neq \mathbf{0}$. When $\mathbf{w}^*$ and $\mathbf{w}$ are both non-zero, $\mathbf{x}^\top\mathbf{w}^* = 0$ and $\mathbf{x}^\top\mathbf{w} = 0$ are measure zero, then

$$\mathcal{E}(\mathbf{w}) - \mathcal{E}(\mathbf{w}^*) = \mathbb{E}_\mathbf{x}|2p^*_\mathbf{x} - 1|\,\mathbf{1}[\mathbf{x}^\top\mathbf{w}\,\mathbf{x}^\top\mathbf{w}^* \leq 0],$$

which is always non-negative. Therefore $\mathbf{w}^* \in \arg\min \mathcal{E}(\mathbf{w})$. This completes the proof of claim A.

For claim B:

$$\mathcal{E}(\mathbf{w}) - \mathcal{E}(\mathbf{w}^*) = 2\mathbb{E}_\mathbf{x}|p^*_\mathbf{x} - 1/2|\,\mathbf{1}[\mathbf{x}^\top\mathbf{w}\,\mathbf{x}^\top\mathbf{w}^* \leq 0]
\leq 2\mathbb{E}_\mathbf{x}|p^*_\mathbf{x} - p_\mathbf{x}(\mathbf{w})|
\leq 2\sqrt{\mathbb{E}_\mathbf{x}|p^*_\mathbf{x} - p_\mathbf{x}(\mathbf{w})|^2}
\leq 2\sqrt{\tfrac{1}{2}\mathbb{E}_\mathbf{x}\mathrm{KL}(p^*_\mathbf{x} \| p_\mathbf{x}(\mathbf{w}))}
= \sqrt{2}\cdot\sqrt{\mathcal{L}(\mathbf{w}) - \mathcal{L}(\mathbf{w}^*)},$$

where the last inequality is by Pinsker's inequality. This completes claim B.

For claim C, notice that $\mathcal{L}(\mathbf{w}) \geq \ln(2)\cdot\mathcal{E}(\mathbf{w})$ for all $\mathbf{w}$. We lower bound $\mathcal{E}(\mathbf{w}^*)$ by

$$\mathcal{E}(\mathbf{w}^*) = 2\mathbb{E}_{g \sim \mathcal{N}(0,1)} \frac{1}{1+\exp(g\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})}\mathbf{1}[g \geq 0]
= \frac{\sqrt{2}}{\pi}\int_0^\infty \frac{\exp(-g^2/2)}{1+\exp(g\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})}\,\mathrm{d}g
\geq \frac{1}{\sqrt{2\pi}(\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}+1)},$$

where we use the error function bounds (Abramowitz & Stegun, 1965):

$$\frac{1}{x+\sqrt{x^2+2}} \leq e^{x^2}\int_x^\infty e^{-t^2}\,\mathrm{d}t \leq \frac{1}{x+\sqrt{x^2+4/\pi}}, \quad x \geq 0.$$

This completes the proof. $\square$

### A.2. Proof of Lemma 3.3

*Proof.* For $\eta \leq 1/\beta$, we have the descent lemma:

$$\widehat{\mathcal{L}}(\mathbf{w}_{t+1}) \leq \widehat{\mathcal{L}}(\mathbf{w}_t) - \eta\|\nabla\widehat{\mathcal{L}}(\mathbf{w}_t)\|^2 + \frac{\eta^2\beta}{2}\|\nabla\widehat{\mathcal{L}}(\mathbf{w}_t)\|^2 \leq \widehat{\mathcal{L}}(\mathbf{w}_t) - \frac{\eta}{2}\|\nabla\widehat{\mathcal{L}}(\mathbf{w}_t)\|^2.$$

Then for the quadratic potential centered at $\mathbf{u}$:

$$\|\mathbf{w}_{t+1} - \mathbf{u}\|^2 = \|\mathbf{w}_t - \mathbf{u}\|^2 + 2\eta\langle\nabla\widehat{\mathcal{L}}(\mathbf{w}_t), \mathbf{u} - \mathbf{w}_t\rangle + \eta^2\|\nabla\widehat{\mathcal{L}}(\mathbf{w}_t)\|^2
\leq \|\mathbf{w}_t - \mathbf{u}\|^2 + 2\eta\left(\widehat{\mathcal{L}}(\mathbf{u}) - \widehat{\mathcal{L}}(\mathbf{w}_{t+1})\right),$$

where the inequality uses convexity and the descent lemma. Telescoping and rearranging:

$$\frac{\|\mathbf{w}_t - \mathbf{u}\|^2}{2\eta t} + \frac{1}{t}\sum_{k=1}^t \widehat{\mathcal{L}}(\mathbf{w}_k) \leq \widehat{\mathcal{L}}(\mathbf{u}) + \frac{\|\mathbf{w}_0 - \mathbf{u}\|^2}{2\eta t}.$$

Using the descent lemma and $\mathbf{w}_0 = \mathbf{0}$:

$$\frac{\|\mathbf{w}_t - \mathbf{u}\|^2}{2\eta t} + \widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{u}) + \frac{\|\mathbf{u}\|^2}{2\eta t}.$$

$\square$

---

## Appendix B. Upper Bounds for Early-Stopped GD

### B.1. Proof of Theorem 3.1

**Lemma B.1.** *Let $\beta := C_0(1 + \mathrm{tr}(\boldsymbol{\Sigma}) + \lambda_1\ln(1/\delta)/n)$, where $C_0 > 1$ is a sufficiently large constant. Assume that $\eta \leq 1/\beta$ and $t$ is such that $\widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1})$. Then with probability at least $1-\delta$, we have $\|\mathbf{w}_t - \mathbf{w}^*_{0:k}\| \leq 1 + \|\mathbf{w}^*_{0:k}\|$.*

*Proof.* We first show that with probability at least $1-\delta$:

$$\|\nabla\widehat{\mathcal{L}}(\mathbf{w})\| \leq \sqrt{\beta}, \quad \|\nabla^2\widehat{\mathcal{L}}(\mathbf{w})\| \leq \beta.$$

This is because

$$\|\nabla\widehat{\mathcal{L}}(\mathbf{w})\| = \left\|\frac{1}{n}\sum_{i=1}^n \ell'(y_i\mathbf{x}_i^\top\mathbf{w})y_i\mathbf{x}_i\right\| \leq \frac{1}{n}\sum_{i=1}^n\|\mathbf{x}_i\| \leq \sqrt{\frac{1}{n}\sum_{i=1}^n\|\mathbf{x}_i\|^2},$$

$$\|\nabla^2\widehat{\mathcal{L}}(\mathbf{w})\| = \left\|\frac{1}{n}\sum_{i=1}^n\ell''(y_i\mathbf{x}_i^\top\mathbf{w})\mathbf{x}_i\mathbf{x}_i^\top\right\| \leq \frac{1}{n}\sum_{i=1}^n\|\mathbf{x}_i\|^2.$$

By Assumption 1 and Bernstein's inequality, with probability at least $1-\delta$:

$$\sum_{i=1}^n\|\mathbf{x}_i\|^2 = \sum_{i=1}^n\sum_j \lambda_j z_{ij}^2 \leq C_0\!\left(n\,\mathrm{tr}(\boldsymbol{\Sigma}) + \lambda_1\ln(1/\delta)\right) \leq \beta.$$

So $\widehat{\mathcal{L}}$ is $\beta$-smooth and $\sqrt{\beta}$-Lipschitz. By the stopping criterion $\widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1})$. Applying Lemma 3.3 to $\mathbf{w}_{t-1}$ with $\mathbf{u} = \mathbf{w}^*_{0:k}$:

$$\|\mathbf{w}_{t-1} - \mathbf{w}^*_{0:k}\|^2 \leq 2\eta(t-1)\!\left(\widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}) - \widehat{\mathcal{L}}(\mathbf{w}_{t-1})\right) + \|\mathbf{w}^*_{0:k}\|^2 \leq \|\mathbf{w}^*_{0:k}\|^2,$$

so $\|\mathbf{w}_{t-1} - \mathbf{w}^*_{0:k}\| \leq \|\mathbf{w}^*_{0:k}\|$. Then by Lipschitzness:

$$\|\mathbf{w}_t - \mathbf{w}^*_{0:k}\| \leq \|\mathbf{w}_t - \mathbf{w}_{t-1}\| + \|\mathbf{w}_{t-1} - \mathbf{w}^*_{0:k}\| \leq \eta\sqrt{\beta} + \|\mathbf{w}^*_{0:k}\| \leq \frac{1}{\sqrt{\beta}} + \|\mathbf{w}^*_{0:k}\| \leq 1 + \|\mathbf{w}^*_{0:k}\|.$$

$\square$

**Lemma B.2.** *Let $\mathbf{w}^* \in \arg\min\mathcal{L}(\mathbf{w})$. Then for every $\mathbf{w}$:*

$$\mathcal{L}(\mathbf{w}) \leq \mathcal{L}(\mathbf{w}^*) + \frac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_{\boldsymbol{\Sigma}}^2.$$

*Proof.* Notice that

$$\nabla^2\mathcal{L}(\mathbf{w}) = \mathbb{E}\,\ell''(y\mathbf{x}^\top\mathbf{w})\mathbf{x}\mathbf{x}^\top = \mathbb{E}\frac{\mathbf{x}\mathbf{x}^\top}{(1+\exp(\mathbf{x}^\top\mathbf{w}))(1+\exp(-\mathbf{x}^\top\mathbf{w}))} \preceq \mathbb{E}\mathbf{x}\mathbf{x}^\top = \boldsymbol{\Sigma}.$$

Moreover, $\nabla\mathcal{L}(\mathbf{w}^*) = \mathbf{0}$. By the midpoint theorem, there exists $\mathbf{v}$ such that:

$$\mathcal{L}(\mathbf{w}) - \mathcal{L}(\mathbf{w}^*) = \langle\nabla\mathcal{L}(\mathbf{w}^*), \mathbf{w}-\mathbf{w}^*\rangle + \frac{1}{2}(\mathbf{w}-\mathbf{w}^*)^\top\nabla^2\mathcal{L}(\mathbf{v})(\mathbf{w}-\mathbf{w}^*) \leq \frac{1}{2}\|\mathbf{w}-\mathbf{w}^*\|_{\boldsymbol{\Sigma}}^2.$$

$\square$

**Lemma B.3.** *Let $C_1 > 1$ be a sufficiently large constant. Then with probability at least $1-\delta$:*

$$\sup_{\|\mathbf{w}\| \leq W} |\mathcal{L}(\mathbf{w}) - \widehat{\mathcal{L}}(\mathbf{w})| \leq C_1 W\sqrt{\frac{(1+\mathrm{tr}(\boldsymbol{\Sigma}))\ln(n/\delta)\ln(1/\delta)}{n}}.$$

*Proof.* This follows from standard Rademacher complexity arguments for 1-Lipschitz loss and linear function class (Bartlett & Mendelson, 2002; Kakade et al., 2008). Let $\tilde{\mathbf{x}} := \mathbf{x}\mathbf{1}[\|\mathbf{x}\| \leq X]$ where $X^2 = C(1+\mathrm{tr}(\boldsymbol{\Sigma}))\ln(n/\delta)$. By Corollary 4 in (Kakade et al., 2008) and Lemma B.9 (below), combining with a union bound gives the result. $\square$

*Proof of Theorem 3.1.* By the stopping condition and Lemma B.1, with probability at least $1-\delta$:

$$\widehat{\mathcal{L}}(\mathbf{w}_t) \leq \widehat{\mathcal{L}}(\mathbf{w}^*_{0:k}), \quad \|\mathbf{w}_t - \mathbf{w}^*_{0:k}\| \leq 1 + \|\mathbf{w}^*_{0:k}\|.$$

Let $W := 1 + 2\|\mathbf{w}^*_{0:k}\|$. Then by Lemmas B.2 and B.3, with probability at least $1-\delta$:

$$\mathcal{L}(\mathbf{w}_t) - \mathcal{L}(\mathbf{w}^*) \leq C_1 W\sqrt{\frac{(1+\mathrm{tr}(\boldsymbol{\Sigma}))\ln(n/\delta)\ln(1/\delta)}{n}} + 0 + C_1 W\sqrt{\frac{(1+\mathrm{tr}(\boldsymbol{\Sigma}))\ln(n/\delta)\ln(1/\delta)}{n}} + \frac{1}{2}\|\mathbf{w}^*_{k:\infty}\|_{\boldsymbol{\Sigma}}^2$$

$$\leq C(1+\|\mathbf{w}^*_{0:k}\|)\sqrt{\frac{(1+\mathrm{tr}(\boldsymbol{\Sigma}))\ln(n/\delta)\ln(1/\delta)}{n}} + \frac{1}{2}\|\mathbf{w}^*_{k:\infty}\|_{\boldsymbol{\Sigma}}^2.$$

$\square$

### B.2. Proof of Theorem 3.2

Throughout this subsection, define

$$W := 1+2\|\mathbf{w}^*\|, \quad X^2 := 2\,\mathrm{tr}(\boldsymbol{\Sigma}) + 2C(1+\lambda_1)\ln\!\left(4C_1(1+\lambda_1^{3/2}W^3)n\sqrt{\mathrm{tr}(\boldsymbol{\Sigma})}/\delta\right), \quad L := WX,$$

$$B := \frac{4C_1(1+\lambda_1^{3/2}W^3)}{W^2X^2} \lesssim \sqrt{\lambda_1}W,$$

where $C_0, C_1 > 1$ are sufficiently large constants. We use tools from (Bartlett et al., 2005). Consider the clipped random variable, function class, and loss:

$$\tilde{\mathbf{x}} := \mathbf{x}\mathbf{1}[\|\mathbf{x}\| \leq X], \quad \mathcal{F} := \left\{\tilde{\mathbf{x}} \mapsto \frac{\mathbf{w}^\top\tilde{\mathbf{x}}}{WX} : \|\mathbf{w}\| \leq W\right\}, \quad \tilde{\ell} : t \mapsto \ell(WXt).$$

**Lemma B.4 (Corollary 5.3 in Bartlett et al., 2005).** *Let $\mathcal{F}$ be a class of functions with ranges in $[-1,1]$ and let $\tilde{\ell}$ be a loss function satisfying: (1) there exists $f^* \in \mathcal{F}$ achieving $\inf_{f\in\mathcal{F}} \mathbb{E}\tilde{\ell}(yf(\tilde{\mathbf{x}}))$; (2) $\tilde{\ell}$ is $L$-Lipschitz; (3) there exists $B \geq 1$ such that $\mathbb{E}(f(\tilde{\mathbf{x}}) - f^*(\tilde{\mathbf{x}}))^2 \leq B(\mathbb{E}\tilde{\ell}(yf(\tilde{\mathbf{x}})) - \mathbb{E}\tilde{\ell}(yf^*(\tilde{\mathbf{x}})))$ for all $f \in \mathcal{F}$. Let $\hat{f} \in \mathcal{F}$ satisfy the empirical stopping criterion. Let $\psi$ be a sub-root function for which $\psi(r) \geq BL\mathbb{E}\mathcal{R}_n\{f \in \mathcal{F} : L^2\mathbb{E}(f(\tilde{\mathbf{x}})-f^*(\tilde{\mathbf{x}}))^2 \leq r\}$. Then for any $r \geq \psi(r)$, with probability at least $1-\delta$:*

$$\mathbb{E}\tilde{\ell}(y\hat{f}(\tilde{\mathbf{x}})) - \mathbb{E}\tilde{\ell}(yf^*(\tilde{\mathbf{x}})) \leq 705\frac{r}{B} + \frac{(11L+27B)\ln(1/\delta)}{n}.$$

**Lemma B.5 (Theorem 6.5 in Bartlett et al., 2005).** *We have*

$$\mathbb{E}\mathcal{R}_n\{f \in \mathcal{F} : L^2\mathbb{E}(f(\tilde{\mathbf{x}})-f^*(\tilde{\mathbf{x}}))^2 \leq r\} \leq \sqrt{\frac{2k}{nL^2}\cdot r + \frac{\sum_{i>k}\lambda_i}{nX^2}},$$

*where $k$ is an arbitrary index.*

**Lemma B.6.** *There exists constant $C > 1$ such that for every $X^2 \geq 2\,\mathrm{tr}(\boldsymbol{\Sigma}) + C(1+\lambda_1)t$:*

$$\Pr(\|\mathbf{x}\| \geq X) \leq \exp(-t), \quad t > 1.$$

*In addition, for $t \geq 2\ln(4C_1(1+\lambda_1^{3/2}W^3))$:*

$$\mathbb{E}\mathbf{x}\mathbf{x}^\top\mathbf{1}[\|\mathbf{x}\| > X] \preceq \frac{1}{2C_1(1+\lambda_1^{3/2}W^3)}\boldsymbol{\Sigma} \preceq \frac{1}{2}\boldsymbol{\Sigma},$$

*which implies $\mathbb{E}\tilde{\mathbf{x}}\tilde{\mathbf{x}}^\top \geq 0.5\boldsymbol{\Sigma}$.*

**Lemma B.7 (from Chardon et al., 2024).** *There exists a constant $C_1 > 0$ such that for every $\mathbf{w}$:*

$$\mathbb{E}\frac{1}{1+\exp(\mathbf{x}^\top\mathbf{w})}\frac{1}{1+\exp(-\mathbf{x}^\top\mathbf{w})}\mathbf{x}\mathbf{x}^\top \succeq \frac{1}{C_1(1+\|\mathbf{w}\|_{\boldsymbol{\Sigma}}^3)}\boldsymbol{\Sigma}.$$

**Lemma B.8.** *Let $X^2 \geq 2\,\mathrm{tr}(\boldsymbol{\Sigma}) + 2C(1+\lambda_1)\ln(4C_1(1+\lambda_1^{3/2}W^3))$ and $B \geq 4C_1(1+\lambda_1^{3/2}W^3)/(W^2X^2)$. Then for every $\|\mathbf{w}\| \leq W$:*

$$\mathbb{E}(\tilde{\mathbf{x}}^\top\mathbf{w} - \tilde{\mathbf{x}}^\top\mathbf{w}^*)^2 \leq W^2X^2B\!\left(\mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w}) - \mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w}^*)\right).$$

*Proof.* Since $\mathbb{E}\tilde{\mathbf{x}}\tilde{\mathbf{x}}^\top \preceq \boldsymbol{\Sigma}$, it suffices to show $\|\mathbf{w}-\mathbf{w}^*\|_{\boldsymbol{\Sigma}}^2 \leq W^2X^2B(\mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w}) - \mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w}^*))$. By the midpoint theorem and the fact that $\mathbf{w}^*$ minimizes $\mathbb{E}_y\ell(y\tilde{\mathbf{x}}^\top\mathbf{w})$, there exists $\mathbf{v}$ between $\mathbf{w}$ and $\mathbf{w}^*$ (so $\|\mathbf{v}\| \leq W$) such that

$$\mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w}) - \mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w}^*) = \frac{1}{2}\mathbb{E}\ell''(y\tilde{\mathbf{x}}^\top\mathbf{v})\langle\tilde{\mathbf{x}}^{\otimes 2}, (\mathbf{w}-\mathbf{w}^*)^{\otimes 2}\rangle.$$

It then suffices to show $\mathbb{E}\ell''(y\tilde{\mathbf{x}}^\top\mathbf{v})\tilde{\mathbf{x}}\tilde{\mathbf{x}}^\top \succeq \frac{2}{W^2X^2B}\boldsymbol{\Sigma}$, which follows from Lemmas B.6 and B.7. $\square$

**Lemma B.9.** *Let $X^2 \geq 2\,\mathrm{tr}(\boldsymbol{\Sigma}) + C(1+\lambda_1)\cdot 2\ln(n\sqrt{\mathrm{tr}(\boldsymbol{\Sigma})})$. Then for every $\mathbf{w}$ with $\|\mathbf{w}\| \leq W$:*

$$\left|\mathcal{L}(\mathbf{w}) - \mathbb{E}\ell(y\tilde{\mathbf{x}}^\top\mathbf{w})\right| \leq \frac{W}{n}.$$

**Lemma B.10.** *Let $\beta := C_0(1+\mathrm{tr}(\boldsymbol{\Sigma}) + \lambda_1\ln(1/\delta)/n)$, $\eta \leq 1/\beta$, and $t$ such that $\widehat{\mathcal{L}}(\mathbf{w}^*) \leq \widehat{\mathcal{L}}(\mathbf{w}_{t-1})$. Then with probability at least $1-\delta$: $\|\mathbf{w}_t - \mathbf{w}^*\| \leq 1 + \|\mathbf{w}^*\|$.*

*Proof of Theorem 3.2.* It is clear that $\tilde{\ell}$ is $L$-Lipschitz and $f^* := \langle\mathbf{w}^*, \cdot\rangle/(WX) \in \arg\inf_{f\in\mathcal{F}}\mathbb{E}\tilde{\ell}(yf(\tilde{\mathbf{x}}))$. By Lemma B.8, condition (3) of Lemma B.4 holds. Let $\hat{f} := \langle\mathbf{w}_t, \cdot\rangle/(WX)$; by Lemma B.10, $\hat{f} \in \mathcal{F}$, and the stopping criterion implies the empirical condition. Applying Lemma B.4 with sub-root function from Lemma B.5, choosing

$$r = \frac{4kB^2}{n} + \sqrt{\frac{2B^2L^2\sum_{i>k}\lambda_i}{nX^2}},$$

we get

$$\mathbb{E}\tilde{\ell}(y\hat{f}(\tilde{\mathbf{x}})) - \mathbb{E}\tilde{\ell}(yf^*(\tilde{\mathbf{x}})) \lesssim \frac{Bk}{n} + \frac{L}{X}\sqrt{\frac{\sum_{i>k}\lambda_i}{n}} + \frac{(L+B)\ln(1/\delta)}{n}.$$

Applying Lemma B.9, substituting the values of $B$, $L$, $X$, and simplifying under $\|\mathbf{w}^*\| \gtrsim 1$, $\lambda_1 \lesssim 1$, $\mathrm{tr}(\boldsymbol{\Sigma}) \gtrsim 1$ yields Theorem 3.2. $\square$

---

## Appendix C. Lower Bounds for Interpolating Estimators

### C.1. Proof of Theorem 4.1

*Proof.* Consider a sequence $(\mathbf{w}_t)_{t \geq 0}$ with $\|\mathbf{w}_t\| \to \infty$ and $\mathbf{w}_t/\|\mathbf{w}_t\| \to \tilde{\mathbf{w}}$. Fix a small constant $\gamma > 0$. Define the event

$$\mathcal{F} := \{\mathbf{x} : |\mathbf{x}^\top\mathbf{w}^*| \leq 10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}},\; \|\mathbf{x}\| \leq 10\sqrt{\mathrm{tr}(\boldsymbol{\Sigma})},\; |\mathbf{x}^\top\tilde{\mathbf{w}}| \geq \gamma\}.$$

Since $\mathbf{x} \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma})$ and $\|\tilde{\mathbf{w}}\|_{\boldsymbol{\Sigma}} > 0$, we have $\Pr(\mathcal{F}) > 0$. Let $t_0$ be such that $\|\mathbf{w}_t/\|\mathbf{w}_t\| - \tilde{\mathbf{w}}\| \leq \gamma/(20\sqrt{\mathrm{tr}(\boldsymbol{\Sigma})})$ for all $t \geq t_0$. Then for every $\mathbf{x} \in \mathcal{F}$ and $t \geq t_0$:

$$\frac{|\mathbf{x}^\top\mathbf{w}_t|}{\|\mathbf{w}_t\|} \geq |\mathbf{x}^\top\tilde{\mathbf{w}}| - \|\mathbf{x}\|\cdot\left\|\frac{\mathbf{w}_t}{\|\mathbf{w}_t\|} - \tilde{\mathbf{w}}\right\| \geq \gamma - 10\sqrt{\mathrm{tr}(\boldsymbol{\Sigma})}\cdot\frac{\gamma}{20\sqrt{\mathrm{tr}(\boldsymbol{\Sigma})}} \geq \frac{\gamma}{2}.$$

**Logistic risk.** For $\mathbf{x} \in \mathcal{F}$ and $t \geq t_0$:

$$\mathcal{L}(\mathbf{w}_t) \geq \mathbb{E}\frac{\ln(1+\exp(|\mathbf{x}^\top\mathbf{w}_t|))}{1+\exp(|\mathbf{x}^\top\mathbf{w}^*|)}\mathbf{1}[\mathbf{x} \in \mathcal{F}]
\geq \mathbb{E}\frac{\|\mathbf{w}_t\|\gamma/2}{1+\exp(10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})}\mathbf{1}[\mathbf{x} \in \mathcal{F}]
\geq \frac{\|\mathbf{w}_t\|\gamma/2}{1+\exp(10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})}\Pr(\mathcal{F}) \to \infty.$$

**Calibration error.** Under event $\mathcal{F}$:

$$p(\mathbf{w}^*;\mathbf{x}) \in \left[\exp(-10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}),\; 1-\exp(-10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})\right].$$

For $t > t_0$ and $\mathbf{x} \in \mathcal{F}$:

$$|p(\mathbf{w}_t;\mathbf{x}) - p(\mathbf{w}^*;\mathbf{x})| \geq \exp(-10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}) - \exp(-\gamma\|\mathbf{w}_t\|/2) \to \exp(-10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}).$$

Therefore

$$\mathcal{C}(\mathbf{w}_t) \geq \left(\exp(-10\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}) - \exp(-\gamma\|\mathbf{w}_t\|/2)\right)^2\Pr(\mathcal{F}) \to \exp(-20\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})\Pr(\mathcal{F}) > 0.$$

$\square$

### C.2. Proof of Theorem 4.2

**Lemma C.1.** *For a non-zero vector $\mathbf{w}$, let $\theta$ be the angle between $\boldsymbol{\Sigma}^{1/2}\mathbf{w}$ and $\boldsymbol{\Sigma}^{1/2}\mathbf{w}^*$. Then*

$$\mathcal{E}(\mathbf{w}) - \mathcal{E}(\mathbf{w}^*) \geq \begin{cases} \dfrac{1}{4\sqrt{2\pi}\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}} & \pi/2 < \theta \leq \pi, \\[8pt] \dfrac{\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}}{48\pi\max\{1,\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}^3\}}(1-\cos\theta) & 0 \leq \theta \leq \pi/2. \end{cases}$$

*Proof.* Let $s(t) = 1/(1+e^{-t})$. Note that for $t > 0$: $s(t) - 1/2 \geq \frac{t}{8}\mathbf{1}[0 < t < 1]$.

Since $\mathbf{x}$ and $-\mathbf{x}$ are identically distributed:

$$\Pr(y\mathbf{x}^\top\mathbf{w} \leq 0) - \Pr(y\mathbf{x}^\top\mathbf{w}^* \leq 0) = 2\mathbb{E}\mathbf{1}[\mathbf{x}^\top\mathbf{w} < 0, \mathbf{x}^\top\mathbf{w}^* > 0](s(\mathbf{x}^\top\mathbf{w}^*) - 1/2) \geq \frac{1}{4}\mathbb{E}\mathbf{x}^\top\mathbf{w}^*\mathbf{1}[\mathbf{x}^\top\mathbf{w} < 0,\; 0 < \mathbf{x}^\top\mathbf{w}^* < 1].$$

Without loss of generality assume $\|\mathbf{w}\|_{\boldsymbol{\Sigma}} = 1$. Write $\boldsymbol{\Sigma}^{1/2}\mathbf{w} = \boldsymbol{\Sigma}^{1/2}\mathbf{w}^*/\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}\cos\theta - \mathbf{v}_\perp\sin\theta$ where $\mathbf{v}_\perp$ is a unit vector orthogonal to $\boldsymbol{\Sigma}^{1/2}\mathbf{w}^*$. Since $\mathbf{x} \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma})$:

$$\Pr(y\mathbf{x}^\top\mathbf{w} \leq 0) - \Pr(y\mathbf{x}^\top\mathbf{w}^* \leq 0) \geq \frac{\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}}{8\pi}\underbrace{\int_{0<g_1<1/\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}},\; g_1\cos\theta < g_2\sin\theta} g_1\exp(-(g_1^2+g_2^2)/2)\,\mathrm{d}g_1\,\mathrm{d}g_2}_{\diamondsuit}.$$

**Case $\pi/2 < \theta \leq \pi$:** $\cos\theta < 0$, so $\diamondsuit \geq \sqrt{2\pi}(1-\exp(-1/\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}^2)) \geq \sqrt{2\pi}/\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}^2$, giving the first bound.

**Case $0 \leq \theta \leq \pi/2$:** Changing to polar coordinates:

$$\diamondsuit \geq \int_0^\theta \sin\psi\left(\frac{1}{2}\int_0^{\min\{1,1/\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}\}} r^2\,\mathrm{d}r\right)\mathrm{d}\psi = \frac{1-\cos\theta}{6\max\{1,\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}}^3\}},$$

giving the second bound. $\square$

**Lemma C.2.** *Let $\mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ and $\Pr(y|\mathbf{z}) = s(y\mathbf{z}^\top\mathbf{v}^*)$. Then for any unit vector $\mathbf{v}$:*

$$\Pr(y\mathbf{z}^\top\mathbf{v} < -0.5) \geq \frac{0.25}{1+\exp(\|\mathbf{v}^*\|)}.$$

*Proof.* By direct calculation using the normal CDF $\Phi$:

$$\mathbb{E}\mathbf{1}[y\mathbf{z}^\top\mathbf{v} < -0.5] \geq \frac{1}{1+\exp(\|\mathbf{v}^*\|)}\!\left(1 - (\Phi(0.5)-\Phi(-0.5)) - 2(1-\Phi(1))\right) \geq \frac{0.25}{1+\exp(\|\mathbf{v}^*\|)}.$$

$\square$

**Lemma C.3.** *Let $(y_i, \mathbf{z}_i)_{i=1}^n$ be independent copies of $(y, \mathbf{z})$. Assume $n \geq C(1+\exp(\|\mathbf{v}^*\|))k\ln(k/\delta)$ for a sufficiently large constant $C > 1$. Let $\mathcal{S} := \{\mathbf{v} : \|\mathbf{v}_{0:k}\| = 1, \mathbf{v}_{k:\infty} = \mathbf{0}\}$. Then*

$$\Pr\!\left(\text{for every } \mathbf{v} \in \mathcal{S},\; \#\{i \in [n] : y_i\mathbf{z}_i^\top\mathbf{v} \leq -0.4\} \geq \frac{n/8}{1+\exp(\|\mathbf{v}^*\|)}\right) \geq 1-\delta.$$

*Proof.* For each unit vector $\mathbf{v}$, define $\xi(\mathbf{v}) := \mathbf{1}[y\mathbf{z}^\top\mathbf{v} \leq -0.5]$. By the multiplicative Chernoff bound and Lemma C.2:

$$\Pr\!\left(\sum_{i=1}^n\xi_i(\mathbf{v}) \leq \frac{n/8}{1+\exp(\|\mathbf{v}^*\|)}\right) \leq \exp\!\left(-\frac{n/32}{1+\exp(\|\mathbf{v}^*\|)}\right).$$

Let $\mathcal{C}$ be an $\epsilon$-$\ell_2$-covering of the $k$-dimensional unit sphere $\mathcal{S}$, with $|\mathcal{C}| = \mathcal{O}((1/\epsilon)^k)$. By a union bound and setting $\epsilon = 0.1/(C\sqrt{k\ln(n/\delta)})$, together with the bound $\max_{i \in [n]}\|\mathbf{z}^{(i)}_{0:k}\| \leq C\sqrt{k\ln(n/\delta)}$ (which holds with probability $\geq 1-0.5\delta$), we obtain the result. $\square$

**Lemma C.4.** *Assume $\mathbf{v}^*$ is $k$-sparse and $n \geq C(1+\exp(\|\mathbf{v}^*\|))k\ln(k/\delta)$, $d \geq Cn\ln(n)\ln(1/\delta)$, for a large constant $C > 1$. Then with probability at least $1-\delta$, for every unit $\mathbf{v}$ with $\max_{i\in[n]}y_i\mathbf{z}_i^\top\mathbf{v} < 0$:*

$$1 - \frac{\langle\mathbf{v},\mathbf{v}^*\rangle}{\|\mathbf{v}^*\|} \geq \sqrt{\frac{1}{C(1+\exp(\|\mathbf{v}^*\|))}\cdot\frac{n}{d}}.$$

*Proof.* Let $\mathbf{v}$ be a unit vector with $\max_{i}y_i\mathbf{z}_i^\top\mathbf{v} < 0$. Since $\mathbf{v}^*$ is $k$-sparse, $1 - \langle\mathbf{v},\mathbf{v}^*\rangle/\|\mathbf{v}^*\| \geq 1 - \|\mathbf{v}_{0:k}\|$. Define the set $\mathcal{I} := \{i : y_i\mathbf{z}_i^\top\mathbf{v}_{0:k} \leq -0.4\|\mathbf{v}_{0:k}\|\}$. By Lemma C.3, $|\mathcal{I}| \geq n/(8(1+\exp(\|\mathbf{v}^*\|)))$ with probability $\geq 1-\delta$. According to (Hsu et al., 2021), all $(y_i, \mathbf{z}^{(i)}_{k:\infty})_{i \in \mathcal{I}}$ are support vectors when $d \geq C|\mathcal{I}|\ln(|\mathcal{I}|)\ln(1/\delta)$. The max-margin direction $\hat{\mathbf{v}} := \mathbf{Z}_{k:\infty}(\mathbf{Z}_{k:\infty}\mathbf{Z}_{k:\infty}^\top)^{-1}\mathbf{y}$ gives

$$\max_{\|\mathbf{u}\|=1}\min_{i\in\mathcal{I}}y_i\mathbf{z}^{(i)}_{k:\infty}\mathbf{u} = \frac{1}{\|\hat{\mathbf{v}}\|} \leq \sqrt{\frac{\|\mathbf{Z}_{k:\infty}\mathbf{Z}_{k:\infty}^\top\|_2}{n}} \leq \sqrt{\frac{2d}{|\mathcal{I}|}}.$$

Since for each $i \in \mathcal{I}$ we have $0 < y_i\mathbf{z}_i^\top\mathbf{v} \leq -0.4\|\mathbf{v}_{0:k}\| + y_i\mathbf{z}^{(i)}_{k:\infty}\mathbf{v}_{k:\infty}$, we get

$$0.4\|\mathbf{v}_{0:k}\| \leq \|\mathbf{v}_{k:\infty}\|\sqrt{2d/|\mathcal{I}|}, \quad \text{so } \|\mathbf{v}_{0:k}\| \leq \frac{1}{\sqrt{1+0.4|\mathcal{I}|/(2d)}}.$$

Under the joint of the three events (probability $\geq 1-3\delta$):

$$1 - \frac{\langle\mathbf{v},\mathbf{v}^*\rangle}{\|\mathbf{v}^*\|} \geq 1 - \|\mathbf{v}_{0:k}\| \geq C\sqrt{\frac{|\mathcal{I}|}{d}} \geq C\sqrt{\frac{1}{1+\exp(\|\mathbf{v}^*\|)}\cdot\frac{n}{d}}.$$

$\square$

*Proof of Theorem 4.2.* Define $\mathbf{v} = \boldsymbol{\Sigma}^{1/2}\mathbf{w}$, $\mathbf{v}^* = \boldsymbol{\Sigma}^{1/2}\mathbf{w}^*$, $\mathbf{z} = \boldsymbol{\Sigma}^{-1/2}\mathbf{x}$, $d := \mathrm{rank}(\boldsymbol{\Sigma})$. Without loss of generality $\|\mathbf{v}\| = 1$. Applying Lemma C.4:

$$1 - \cos\theta = 1 - \frac{\langle\mathbf{v},\mathbf{v}^*\rangle}{\|\mathbf{v}^*\|} \gtrsim \sqrt{\frac{n}{d}} \gtrsim \frac{1}{\sqrt{\ln(n)\ln(1/\delta)}}.$$

Applying Lemma C.1 completes the proof. $\square$

---

## Appendix D. Early Stopping and $\ell_2$-Regularization

### D.1. Proof of Theorem 5.1

*Proof.* Apply the $\ell_2$-regularized ERM $\mathbf{u}_\lambda$ with $\lambda = 1/(\eta t)$ as a comparator in Lemma 3.3. By the first-order stationarity condition: $-\nabla\widehat{\mathcal{L}}(\mathbf{u}_\lambda) = \lambda\mathbf{u}_\lambda = \frac{1}{\eta t}\mathbf{u}_\lambda$. Then:

$$\frac{1}{2}\|\mathbf{w}_t - \mathbf{u}_\lambda\|^2 - \frac{1}{2}\|\mathbf{u}_\lambda\|^2 \leq \eta t\!\left(\widehat{\mathcal{L}}(\mathbf{u}_\lambda) - \widehat{\mathcal{L}}(\mathbf{w}_t)\right) \leq \eta t\langle\nabla\widehat{\mathcal{L}}(\mathbf{u}_\lambda), \mathbf{u}_\lambda - \mathbf{w}_t\rangle = -\langle\mathbf{u}_\lambda, \mathbf{u}_\lambda - \mathbf{w}_t\rangle,$$

which is equivalent to

$$\|\mathbf{w}_t - \mathbf{u}_\lambda\|^2 \leq \langle\mathbf{u}_\lambda, \mathbf{w}_t\rangle - \frac{1}{2}\|\mathbf{u}_\lambda\|^2 \quad \Leftrightarrow \quad \|\mathbf{w}_t - \mathbf{u}_\lambda\|^2 \leq \frac{1}{2}\|\mathbf{w}_t\|^2. \tag{3}$$

**Angle bound.** Reformulate (3) as $2\langle\mathbf{u}_\lambda, \mathbf{w}_t\rangle \geq \frac{1}{2}\|\mathbf{w}_t\|^2 + \|\mathbf{u}_\lambda\|^2$. Then:

$$\cos(\mathbf{w}_t, \mathbf{u}_\lambda) = \frac{\langle\mathbf{u}_\lambda,\mathbf{w}_t\rangle}{\|\mathbf{u}_\lambda\|\cdot\|\mathbf{w}_t\|} \geq \frac{\frac{1}{2}(\frac{1}{2}\|\mathbf{w}_t\|^2 + \|\mathbf{u}_\lambda\|^2)}{\|\mathbf{u}_\lambda\|\cdot\|\mathbf{w}_t\|} \geq \frac{1}{\sqrt{2}},$$

using $(a+b)/2 \geq \sqrt{ab}$.

**Norm bounds.** Using triangle inequality with (3):

$$\frac{1}{\sqrt{2}}\|\mathbf{w}_t\| \geq \|\mathbf{w}_t - \mathbf{u}_\lambda\| \geq \begin{cases}\|\mathbf{w}_t\| - \|\mathbf{u}_\lambda\|\\\|\mathbf{u}_\lambda\| - \|\mathbf{w}_t\|\end{cases},$$

which implies $\frac{\sqrt{2}}{\sqrt{2}+1}\|\mathbf{u}_\lambda\| \leq \|\mathbf{w}_t\| \leq \frac{\sqrt{2}}{\sqrt{2}-1}\|\mathbf{u}_\lambda\|$. $\square$

### D.2. Proof of Theorem 5.2

Recall the dataset $(\mathbf{x}_i,y_i)_{i=1}^n$ is linearly separable. Define the margin and max-margin direction:

$$\gamma := \max_{\|\mathbf{w}\|=1}\min_{i\in[n]}y_i\mathbf{x}_i^\top\mathbf{w} > 0, \quad \tilde{\mathbf{w}} := \arg\max_{\|\mathbf{w}\|=1}\min_{i\in[n]}y_i\mathbf{x}_i^\top\mathbf{w}.$$

WLOG assume $\tilde{\mathbf{w}} = \mathbf{e}_1$. Use $(w_t, \bar{\mathbf{w}}_t)$ and $(u_\nu, \bar{\mathbf{u}}_\nu)$ for the GD and regularization paths. Data: $\mathbf{x}_i = (x_i, \bar{\mathbf{x}}_i)$, $y_ix_i \geq \gamma$. Support vectors: $\mathcal{S} := \{i : y_ix_i^\top\tilde{\mathbf{w}} = \gamma\}$. By (Wu et al., 2023), under Assumption 2, the dataset $(\bar{\mathbf{x}}_i, y_i)_{i\in\mathcal{S}}$ is strictly non-separable: there exists $b > 0$ such that for every $\bar{\mathbf{w}}$, there exists $i \in \mathcal{S}$ with $y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{w}} \leq -b\|\bar{\mathbf{w}}\|$. Define $G(\bar{\mathbf{w}}) := \sum_{i\in\mathcal{S}}\exp(-y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{w}})$ and $\bar{\mathbf{w}}^* := \arg\min G(\bar{\mathbf{w}})$, which is finite.

**Lemma D.1 (Wu et al., 2023).** *Let $\mathbf{w}_t := (w_t, \bar{\mathbf{w}}_t)$ be the GD path with any fixed stepsize $\eta > 0$. Under Assumption 2, $w_t$ is increasing and $\lim_{t\to\infty}w_t = \infty$, $\lim_{t\to\infty}\bar{\mathbf{w}}_t = \bar{\mathbf{w}}^*$.*

**Lemma D.2.** *Let $\mathbf{u}_\lambda = (u_\lambda, \bar{\mathbf{u}}_\lambda)$ be the $\ell_2$-regularization path. Under Assumption 2: $\lim_{\lambda\to 0}u_\lambda = \infty$ and $\lim_{\lambda\to 0}\bar{\mathbf{u}}_\lambda = \bar{\mathbf{w}}^*$.*

*Proof.* By the first-order condition:

$$\lambda u_\lambda = \frac{1}{n}\sum_{i=1}^n\frac{y_ix_i}{1+\exp(y_ix_iu_\lambda + y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{u}}_\lambda)}, \quad \lambda\bar{\mathbf{u}}_\lambda = \frac{1}{n}\sum_{i=1}^n\frac{y_i\bar{\mathbf{x}}_i}{1+\exp(y_ix_iu_\lambda + y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{u}}_\lambda)}.$$

**$u_\lambda \to \infty$:** Using $y_ix_i \geq \gamma$ and $G(\bar{\mathbf{u}}_\lambda) \geq 1$:

$$\lambda u_\lambda \geq \frac{\gamma}{2n}\sum_{i\in\mathcal{S}}\exp(-y_ix_iu_\lambda - y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{u}}_\lambda) = \frac{\gamma}{2n}\exp(-\gamma u_\lambda)G(\bar{\mathbf{u}}_\lambda) \geq \frac{\gamma}{2n}\exp(-\gamma u_\lambda),$$

which implies $u_\lambda \to \infty$ as $\lambda \to 0$.

**$\bar{\mathbf{u}}_\lambda$ is bounded:** Using the strict non-separability (there exists $i^* \in \mathcal{N} \cap \mathcal{S}$ with $y_{i^*}\bar{\mathbf{x}}_{i^*}^\top\bar{\mathbf{u}}_\lambda \leq -b\|\bar{\mathbf{u}}_\lambda\|$ where $\mathcal{N} := \{i : y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{u}}_\lambda < 0\}$) and $u_\lambda > 0$:

$$b\|\bar{\mathbf{u}}_\lambda\| \leq n(1+\exp(\gamma u_\lambda))\exp(-\gamma u_\lambda) \leq 2n,$$

showing $\bar{\mathbf{u}}_\lambda$ is bounded. An upper bound on $u_\lambda$ then follows: $\lambda\exp(\gamma u_\lambda) \leq XF_{\max}/(nu_\lambda) \to 0$.

**$\bar{\mathbf{u}}_\lambda \to \bar{\mathbf{w}}^*$:** Taking $\lambda \to 0$ in the equation for $\lambda\bar{\mathbf{u}}_\lambda\exp(\gamma u_\lambda)$ and using $y_ix_i > \gamma$ for $i \notin \mathcal{S}$, $y_ix_i = \gamma$ for $i \in \mathcal{S}$:

$$0 = \lim_{\lambda\to 0}\frac{1}{n}\sum_{i\in\mathcal{S}}\frac{y_i\bar{\mathbf{x}}_i}{\exp(y_i\bar{\mathbf{x}}_i^\top\bar{\mathbf{u}}_\lambda)}.$$

Thus $\bar{\mathbf{u}}_0$ satisfies the first-order stationarity condition of $G(\cdot)$, so $\bar{\mathbf{u}}_0 = \bar{\mathbf{w}}^*$. $\square$

*Proof of Theorem 5.2.* Since $u_\lambda$ is continuous, $u_\infty = 0$, and $\lim_{\lambda\to 0}u_\lambda = \infty$, while $w_0 = 0$, $w_t$ is increasing, and $w_t \to \infty$, for each $t$ we can choose $\lambda(t)$ such that $u_{\lambda(t)} = w_t$ and $\lambda(t) \to 0$. Then:

$$\|\mathbf{u}_{\lambda(t)} - \mathbf{w}_t\| = \|\bar{\mathbf{u}}_{\lambda(t)} - \bar{\mathbf{w}}_t\| \leq \|\bar{\mathbf{u}}_{\lambda(t)} - \bar{\mathbf{w}}^*\| + \|\bar{\mathbf{w}}_t - \bar{\mathbf{w}}^*\| \to 0.$$

$\square$

### D.3. Proof of Theorem 5.3

*Proof.* As the dataset is linearly separable and we only care about the asymptotic, WLOG use the exponential loss instead of the logistic loss, and analyze gradient flow (the argument applies to GD with any fixed stepsize $\eta > 0$).

Denote $\mathbf{w} = (w^{(1)}, w^{(2)})$. The empirical risk is

$$\widehat{\mathcal{L}}(w^{(1)}, w^{(2)}) := \frac{1}{2}\!\left(\exp(-\gamma w^{(1)}) + \exp(-\gamma w^{(1)} - \gamma_2 w^{(2)})\right).$$

It suffices to compare the GF and $\ell_2$-regularization paths under $\ln\widehat{\mathcal{L}}$, where

$$\ln\widehat{\mathcal{L}}(w^{(1)}, w^{(2)}) = -\gamma w^{(1)} + \ln\!\left(1 + \exp(-\gamma_2 w^{(2)})\right) - \ln(2).$$

**GF path.** The ODEs:

$$\mathrm{d}w^{(1)}_t = \gamma\,\mathrm{d}t, \quad \mathrm{d}w^{(2)}_t = \gamma_2\frac{\exp(-\gamma_2 w^{(2)})}{1+\exp(-\gamma_2 w^{(2)})}\,\mathrm{d}t, \quad w^{(1)}_0 = w^{(2)}_0 = 0.$$

Solving: $w^{(1)}_t = \gamma t$ and $\left|w^{(2)}_t - \frac{\ln(1+\gamma_2^2 t)}{\gamma_2}\right| \leq C$ for some constant $C > 0$.

**$\ell_2$-regularization path.** The stationarity conditions:

$$-\gamma + \lambda u^{(1)}_\lambda = 0, \quad -\gamma_2\frac{\exp(-\gamma_2 u^{(2)})}{1+\exp(-\gamma_2 u^{(2)})} + \lambda u^{(2)}_\lambda = 0.$$

Solving: $u^{(1)}_\lambda = \gamma/\lambda$ and $\left|u^{(2)}_\lambda - \frac{\ln(\gamma_2^2/\lambda) - \ln\ln(\gamma_2^2/\lambda)}{\gamma_2}\right| \leq C$ for $\lambda \leq \gamma_2^2/e$.

**Divergence.** As $t \to \infty$ and $\lambda \to 0$, both paths diverge in both coordinates. However, $w^{(1)}_t = \gamma t$ grows linearly while $u^{(1)}_\lambda = \gamma/\lambda$, and $w^{(2)}_t \approx \ln(1+\gamma_2^2 t)/\gamma_2$ grows logarithmically while $u^{(2)}_\lambda \approx \ln(\gamma_2^2/\lambda)/\gamma_2$. These rates are compatible in isolation, but for any matching $\lambda(t)$ (so $u^{(1)}_\lambda = w^{(1)}_t$ would require $\lambda = 1/t$, giving $u^{(2)}_\lambda \approx \ln(t)/\gamma_2$ while $w^{(2)}_t \approx \ln(1+\gamma_2^2 t)/\gamma_2$, which agree up to constants), we have

$$\|{\mathbf{w}}_t - \mathbf{u}_{\lambda(t)}\| \gtrsim \ln\ln(\|\mathbf{w}_t\|) \to \infty.$$

$\square$
