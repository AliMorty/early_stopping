# Findings: Is E[w_tilde] an Unbiased Estimator of w*?

## Setup

We sampled M independent training sets of size n=1000 in d=2000 dimensions
(k=100 informative components), computed the max-margin direction
$\tilde{w}_m$ for each, and averaged them.

We ran two rounds: first M=20, then extended to M=50.

## Key Numbers

| Metric | M=20 | M=50 |
|--------|------|------|
| cos(avg w_tilde, w*) | 0.061 | 0.098 |
| Cosine in informative subspace | 0.607 | 0.767 |
| Norm of avg w_tilde (informative) | 0.022 | 0.018 |
| Norm of avg w_tilde (uninformative) | 0.221 | 0.140 |
| Mean individual cos(w_tilde_m, w*) | 0.014 | 0.014 |
| Std individual cos(w_tilde_m, w*) | 0.007 | 0.009 |

## Observations

**1. Individual w_tilde directions are nearly orthogonal to w*.**
Each individual w_tilde_m has a cosine similarity of only about 0.014 with
w*. In 2000 dimensions, this is barely above what you'd expect from random
vectors. The max-margin direction for any single dataset tells you almost
nothing about w*.

**2. The noise dimensions are cancelling out, but slowly.**
The uninformative norm dropped from 0.221 (M=20) to 0.140 (M=50). This
confirms that the noise components are averaging out as expected, but
M=50 is not nearly enough to get rid of them. The uninformative norm is
still much larger than the informative norm (0.140 vs 0.018), meaning
most of the energy in the average w_tilde is still noise.

**3. Within the informative subspace, the direction is trending toward w*.**
The cosine similarity restricted to the informative subspace improved from
0.607 to 0.767 as M went from 20 to 50. This is a positive sign, but 0.767
is still far from 1.0. It is not yet clear whether this will converge to
1.0 (meaning E[w_tilde] is proportional to w* in the informative subspace)
or to some other value (meaning the eigenvalue structure biases the
direction).

**4. The overall cosine similarity is very low.**
cos(avg w_tilde, w*) = 0.098 at M=50. This is because the uninformative
noise still dominates the average. Even if the informative components
perfectly aligned with w*, the large uninformative component would dilute
the overall cosine similarity.

## Conclusion (preliminary)

The results at M=50 are **inconclusive but encouraging**:
- The trend is in the right direction (both metrics improving with M)
- The noise is cancelling as theory predicts
- But M=50 is not enough to answer the question definitively

To get a clearer answer, we would need M in the hundreds or thousands.
The slow convergence itself is informative: it confirms that w_tilde is
an extremely noisy estimator, regardless of whether it is unbiased.

-- Claude (Opus 4.6)
