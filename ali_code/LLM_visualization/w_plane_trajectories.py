"""
Standalone helpers for visualizing weight trajectories in a 2D plane.

Nothing here touches your notebook or the LogisticRegressionClass. These are
plain functions you can read, run, and then copy (in whole or in part) into
your class as methods. Where a function needs the model (to fit / compute
gradients), pass your instance in as `model`.

Two ways to get a 2D plane:
  1. A plane you choose:   basis_from_vectors(w_1, w_2)
     (either axis can be a w_tilde fitted from some dataset)
  2. Automatic plane:      pca_basis(list_of_w_points)  -> top-2 variance dirs

Both return an orthonormal (d x 2) matrix B. Any weight vector w (length d)
projects to plane coordinates with:   coords = w @ B        # shape (2,)
A whole trajectory (T x d array) projects with:  W @ B      # shape (T, 2)
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# --------------------------------------------------------------------------- #
# 1. Fit w_tilde (logistic-regression minimizer) on a given dataset
# --------------------------------------------------------------------------- #
def find_w_tilde(model, X_data, y_data, n_steps=5000, eta=None, w_0=None):
    """
    Return w_tilde: the logistic-loss minimizer on (X_data, y_data), found by
    running gradient descent to (approximate) convergence.

    Reuses the model's own `grad` and `logistic_loss`, so it stays consistent
    with the rest of your code. If your data is separable the loss keeps
    decreasing and w grows in norm -- `n_steps` then just controls how far
    along the (fixed) direction you go.

    Parameters
    ----------
    model   : your LogisticRegressionClass instance (needs .grad, .eta, .d)
    X_data  : (n, d) design matrix
    y_data  : (n,) labels in {-1, +1}
    n_steps : GD iterations
    eta     : step size (defaults to model.eta)
    w_0     : starting point (defaults to zeros)
    """
    d = X_data.shape[1]
    eta = model.eta if eta is None else eta
    w = np.zeros(d) if w_0 is None else np.array(w_0, dtype=float)

    for _ in range(n_steps):
        w = w - eta * model.grad(w, X_data, y_data)
    return w


def find_w_tilde_svm(X_data, y_data, normalize=False):
    """
    Return w_tilde as the *exact* max-margin (hard-margin SVM) solution:

        min  0.5 * ||w||^2   s.t.   y_i * (x_i . w) >= 1  for all i

    This is the direction logistic GD only *approaches* (at rate ~1/log t),
    so this is far more efficient/accurate than running GD to convergence
    when the data is linearly separable.

    Solved exactly as a QP with cvxpy. If the data is NOT separable the
    constraints are infeasible; cvxpy reports status "infeasible" and this
    raises -- use the logistic `find_w_tilde` there instead.

    normalize : if True, return w / ||w|| (pure direction). By construction
                the unnormalized w has margin exactly 1.
    """
    n, d = X_data.shape
    Xy = X_data * y_data[:, None]                      # rows: y_i * x_i

    w = cp.Variable(d)
    objective = cp.Minimize(0.5 * cp.sum_squares(w))
    constraints = [Xy @ w >= 1]                        # y_i (x_i . w) >= 1
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SVM QP not solved (status={prob.status}); "
                           "is the data linearly separable?")
    w_val = np.asarray(w.value).ravel()
    return w_val / np.linalg.norm(w_val) if normalize else w_val


# --------------------------------------------------------------------------- #
# 1b. Collect the w-sequence of a GD run
# --------------------------------------------------------------------------- #
def collect_w_trajectory(model, X_data, y_data, n_steps, eta=None, w_0=None):
    """
    Re-run GD and return the (n_steps+1, d) array of weights visited.

    Your `run_GD_for_t_steps` records *losses* but not the w's, and the
    plane plots need the w's. This mirrors your GD update exactly
    (w <- w - eta * grad) so the trajectory matches. Later you may prefer to
    have run_GD_for_t_steps append w_current itself and skip this.
    """
    d = X_data.shape[1]
    eta = model.eta if eta is None else eta
    w = np.zeros(d) if w_0 is None else np.array(w_0, dtype=float)

    ws = [w.copy()]
    for _ in range(n_steps):
        w = w - eta * model.grad(w, X_data, y_data)
        ws.append(w.copy())
    return np.array(ws)                        # (n_steps+1, d)


# --------------------------------------------------------------------------- #
# 2. Build an orthonormal 2D basis (the "plane")
# --------------------------------------------------------------------------- #
def basis_from_vectors(w_1, w_2):
    """
    Orthonormal (d x 2) basis B spanning the plane through w_1 and w_2.

    Column 0 points along w_1 exactly; column 1 is the part of w_2 orthogonal
    to w_1 (Gram-Schmidt). So the horizontal axis in the plot is "the w_1
    direction" and the vertical axis is "what's new in w_2".
    """
    w_1 = np.asarray(w_1, dtype=float)
    w_2 = np.asarray(w_2, dtype=float)

    u1 = w_1 / np.linalg.norm(w_1)
    w2_perp = w_2 - (w_2 @ u1) * u1
    norm = np.linalg.norm(w2_perp)
    if norm < 1e-12:
        raise ValueError("w_1 and w_2 are parallel; they don't define a plane.")
    u2 = w2_perp / norm
    return np.column_stack([u1, u2])          # (d, 2)


def pca_basis(w_points, center=True):
    """
    Automatic 2D plane: top-2 principal directions of a cloud of weight
    vectors (e.g. all points of one or several trajectories stacked together).

    w_points : (M, d) array of weight vectors.
    center   : subtract the mean before SVD (usual PCA). Returns (B, mean) so
               you can project with (W - mean) @ B and stay consistent.
    """
    W = np.asarray(w_points, dtype=float)
    mean = W.mean(axis=0) if center else np.zeros(W.shape[1])
    U, S, Vt = np.linalg.svd(W - mean, full_matrices=False)
    B = Vt[:2].T                              # (d, 2), top-2 components
    return B, mean


def project(W, B, mean=None):
    """Project weights W (d,) or (T, d) onto plane B (d,2). Returns (2,) or (T,2)."""
    W = np.asarray(W, dtype=float)
    if mean is not None:
        W = W - mean
    return W @ B


# --------------------------------------------------------------------------- #
# 3. Plot trajectories in the plane
# --------------------------------------------------------------------------- #
def plot_trajectory_over(w_1, w_2, trajectory_dict, basis=None, mean=None,
                         mark_points=None, ax=None, title=None):
    """
    Plot one or more weight trajectories in the plane defined by (w_1, w_2).

    Parameters
    ----------
    w_1, w_2 : d-vectors defining the plane axes. Either can be a w_tilde.
               Ignored if you pass an explicit `basis` (e.g. a PCA basis).
    trajectory_dict : {label: W} where each W is a (T, d) array (or list of
               d-vectors) -- the sequence of weights along that trajectory.
    basis    : optional precomputed (d, 2) B. If None, uses
               basis_from_vectors(w_1, w_2).
    mean     : optional centering vector (use the one pca_basis returned).
    mark_points : optional {label: w} extra single points to scatter
               (e.g. {"w_star": w_star, "w_tilde": w_tilde}).
    ax       : optional matplotlib axis to draw on.

    Returns the axis.
    """
    if basis is None:
        basis = basis_from_vectors(w_1, w_2)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for label, W in trajectory_dict.items():
        W = np.asarray(W, dtype=float)
        coords = project(W, basis, mean)                 # (T, 2)
        line, = ax.plot(coords[:, 0], coords[:, 1], marker="o", ms=2, label=label)
        # mark start (square) and end (star) so direction of travel is clear
        ax.scatter(*coords[0], color=line.get_color(), marker="s", s=40, zorder=3)
        ax.scatter(*coords[-1], color=line.get_color(), marker="*", s=120, zorder=3)

    if mark_points:
        for label, w in mark_points.items():
            p = project(w, basis, mean)
            ax.scatter(*p, marker="X", s=90, zorder=4)
            ax.annotate(label, p, textcoords="offset points", xytext=(5, 5))

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("axis 1  (w_1 direction)")
    ax.set_ylabel("axis 2  (w_2 direction ⟂)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.set_title(title or "Weight trajectories in the (w_1, w_2) plane")
    return ax
