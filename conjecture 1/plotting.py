import numpy as np
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import norm


def load_run(filepath):
    """Load a saved run from a .pkl file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def compute_metrics(data):
    """Compute plottable metrics from raw saved data. Returns a dict of available metrics."""
    w_history = data.get("w_history")
    w_star = data["config"]["w_star"]
    w_tilde = data.get("w_tilde")

    if w_history is None:
        return {}

    w_star_dir = w_star / norm(w_star)

    times = []
    cos_wstar = []
    cos_wtilde = []
    norms = []

    for t, w in w_history:
        if t == 0:
            continue
        w_dir = w / norm(w)
        times.append(t)
        cos_wstar.append(np.dot(w_dir, w_star_dir))
        norms.append(norm(w))
        if w_tilde is not None:
            cos_wtilde.append(np.dot(w_dir, w_tilde))

    metrics = {
        "times": np.array(times),
        "cos_wstar": np.array(cos_wstar),
        "norms": np.array(norms),
        "w_star_norm": norm(w_star),
    }

    if w_tilde is not None:
        metrics["cos_wtilde"] = np.array(cos_wtilde)

    # Empirical loss history
    loss_history = data.get("loss_history")
    if loss_history is not None:
        metrics["loss_times"] = np.array([t for t, l in loss_history if t > 0])
        metrics["loss_values"] = np.array([l for t, l in loss_history if t > 0])

    # Population loss history
    pop_loss_history = data.get("pop_loss_history")
    if pop_loss_history is not None:
        metrics["pop_loss_times"] = np.array([t for t, l in pop_loss_history if t > 0])
        metrics["pop_loss_values"] = np.array([l for t, l in pop_loss_history if t > 0])

    # Stopping times
    stopping_times = data.get("stopping_times")
    if stopping_times is not None and len(stopping_times) > 0:
        metrics["stopping_times"] = stopping_times

    return metrics


def plot_from_file(filepath, save_path=None):
    """Load a .pkl file and plot all available metrics."""
    data = load_run(filepath)
    plot_from_data(data, save_path=save_path)


def plot_from_data(data, save_path=None):
    """Plot all available metrics from a loaded data dict."""
    metrics = compute_metrics(data)
    config = data["config"]

    if not metrics:
        print("No plottable data found.")
        return

    times = metrics["times"]
    stopping_times = metrics.get("stopping_times")

    # Determine which plots we can make
    available = []
    if "norms" in metrics:
        available.append("norm")
    if "loss_values" in metrics or "pop_loss_values" in metrics:
        available.append("loss")
    if "cos_wstar" in metrics:
        available.append("angle")

    if not available:
        print("No metrics available to plot.")
        return

    n_plots = len(available)
    ncols = min(n_plots, 2)
    nrows = (n_plots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    title = f"n={config['n']}, d={config['d']}, k={config['k']}, seed={config['seed']}"
    fig.suptitle(title, fontsize=13)

    plot_idx = 0

    def add_stopping_lines(ax):
        if stopping_times is not None:
            for st in stopping_times:
                ax.axvline(st, color='red', linestyle='--', alpha=0.4, linewidth=1)
            ax.axvline(stopping_times[0], color='red', linestyle='--', alpha=0.6,
                       linewidth=1.5, label=f'stopping time (first={stopping_times[0]})')

    # Norm
    if "norm" in available:
        ax = axes[plot_idx]
        ax.semilogx(times, metrics["norms"], linewidth=2, color='green')
        ax.axhline(metrics["w_star_norm"], color='gray', linestyle='--', alpha=0.7,
                   linewidth=1.5, label=rf'$\|w^*\| = {metrics["w_star_norm"]:.2f}$')
        add_stopping_lines(ax)
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel(r'$\|w_t\|$')
        ax.set_title('Norm of GD iterates')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Loss (training + population on same axes)
    if "loss" in available:
        ax = axes[plot_idx]
        if "loss_values" in metrics:
            ax.semilogx(metrics["loss_times"], metrics["loss_values"],
                         linewidth=2, color='red', label='Training loss')
        if "pop_loss_values" in metrics:
            ax.semilogx(metrics["pop_loss_times"], metrics["pop_loss_values"],
                         linewidth=2, color='purple', label='Population loss')
        add_stopping_lines(ax)
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel('Logistic loss')
        ax.set_title('Training vs Population loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Angles
    if "angle" in available:
        ax = axes[plot_idx]
        angle_wstar = np.degrees(np.arccos(np.clip(metrics["cos_wstar"], -1, 1)))
        ax.semilogx(times, angle_wstar,
                     label=r'Angle to $w^*/\|w^*\|$', linewidth=2)
        if "cos_wtilde" in metrics:
            angle_wtilde = np.degrees(np.arccos(np.clip(metrics["cos_wtilde"], -1, 1)))
            ax.semilogx(times, angle_wtilde,
                         label=r'Angle to $\tilde{w}$', linewidth=2)
        add_stopping_lines(ax)
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Angular distance to targets')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
