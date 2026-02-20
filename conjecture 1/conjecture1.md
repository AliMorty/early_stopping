I want to use the basic setup for $\eta$ and $n$ and $d$ and set $T=10,000$ for now. in overparametrized regimer we have $d \gg n$. 


**Data Model.** We mainly focus on a *well-specified* setting formalized by the following conditions. However, part of our results can also be applied to misspecified cases.

**Assumption 1** (Well-specification). *Let* $\Sigma \in \mathbb{H}^{\otimes 2}$ *be positive semi-definite (PSD) and* $\text{tr}(\Sigma) < \infty$. *Let* $\mathbf{w}^* \in \mathbb{H}$ *be such that* $\|\mathbf{w}^*\|_\Sigma < \infty$. *Assume that* $(\mathbf{x}, y) \in \mathbb{H} \otimes \{\pm 1\}$ *is given by*

$$\mathbf{x} \sim \mathcal{N}(0, \Sigma), \quad \Pr(y | \mathbf{x}) = \frac{1}{1 + \exp(-y \mathbf{x}^\top \mathbf{w}^*)}.$$

Generate a fix dataset of $n$ points as explained above for our experiment. 

run gradient descent for $T$ rounds and store $w_1, \ldots, w^T$. and 
the goal is to see whether $\frac{w_t}{||w_t||}$ 
converges to $\tilde{w}$ or $w^*$

the goal is to plot but I don't know what the best plot is when we want to compare vectors in high dimensional space. 