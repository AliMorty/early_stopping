import numpy as np
import pickle
import os
from datetime import datetime


def power_law_config(d, k):
    """Default config: eigenvalues = i^{-2}, w* = first k components = 1."""
    eigenvalues = np.array([(i + 1) ** (-2) for i in range(d)])
    w_star = np.zeros(d)
    w_star[:k] = 1.0
    return eigenvalues, w_star


def theoretical_eta(eigenvalues, n, C0=2.0, delta=0.01):
    """Step size upper bound from Theorem 3.1."""
    tr_sigma = np.sum(eigenvalues)
    lambda_1 = eigenvalues[0]
    return 1.0 / (C0 * (1 + tr_sigma + lambda_1 * np.log(1.0 / delta) / n))


def create_model(n, d, k, eigenvalues=None, w_star=None, eta=None, seed=0):
    """Convenience function: fills in defaults and returns a model."""
    from model import OverparameterizedLogisticRegression

    if eigenvalues is None and w_star is None:
        eigenvalues, w_star = power_law_config(d, k)
    elif eigenvalues is None or w_star is None:
        raise ValueError("Provide both eigenvalues and w_star, or neither.")

    if eta is None:
        eta = theoretical_eta(eigenvalues, n)

    return OverparameterizedLogisticRegression(
        n=n, d=d, k=k, eigenvalues=eigenvalues, w_star=w_star, eta=eta, seed=seed
    )


def run_and_save(model, save_dir="results"):
    """Run GD is assumed done. Bundles raw data + config into a .pkl file."""
    os.makedirs(save_dir, exist_ok=True)

    data = {
        "config": {
            "n": model.n,
            "d": model.d,
            "k": model.k,
            "seed": model.seed,
            "eigenvalues": model.eigenvalues,
            "w_star": model.w_star,
            "eta": model.eta,
            "num_iterations": model.t_current,
        },
        "w_init": np.zeros(model.d),
        "w_history": model.w_history,
        "loss_history": model.loss_history,
        "w_tilde": model.w_tilde,
        "stopping_times": model.stopping_times,
        "pop_loss_history": model.pop_loss_history if model.pop_loss_history else None,
        "timestamp": datetime.now().isoformat(),
    }

    filename = f"run_n{model.n}_d{model.d}_k{model.k}_seed{model.seed}.pkl"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved to {filepath}")
    return filepath
