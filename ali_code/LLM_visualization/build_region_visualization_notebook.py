"""One-off generator for region_visualization.ipynb.

Interactive plotly view of the region-of-trajectories experiment
(LLM_generated_region_experiment_v1), with GD *time* encoded along each
projected trajectory via:
  1. equal-step tick markers  -> their spatial density shows dwell time
                                 (many steps per unit length = slow = lots of time here)
  2. color by log10(step)     -> smooth hue from early (dark) to late (bright)
  3. hover                    -> point at any marker to read the exact GD step number

Regenerate: python ali_code/LLM_visualization/build_region_visualization_notebook.py
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "region_visualization.ipynb"

cell_imports = '''\
import os, sys
# logistic_regression.py (Ali's nbconvert'd algorithm) lives in ali_code/, the parent of this LLM_visualization/ folder
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import plotly.graph_objects as go

from logistic_regression import LogisticRegressionClass
'''

cell_md_intro = '''\
# Region-of-trajectories, with *time* along the line

Each GD trajectory is projected onto the 2D plane
(axis 1 = `w*`, axis 2 = `sum_i w_tilde_i`) by `LLM_generated_region_experiment_v1`.

A plain line hides **how many steps** were spent in each part of the path. For logistic
GD on separable data `||w_t|| ~ log(t)`, so equal-size spatial moves take
exponentially more steps as `t` grows: ~90% of a 100k-step run is crammed into a tiny
end-segment. This notebook makes that visible:

- **equal-step tick markers** — dense clusters = many steps per unit length = lots of time spent there.
- **color = log10(step)** — dark (early) → bright (late).
- **hover** — point at any marker to read the exact step number. Zoom into the crammed
  end-blob and hover to drill in.
'''

cell_params = '''\
# --- experiment setup (edit freely) ---
n = 100
d = 200
k = 10
number_of_trajectories = 3
t_steps = int(1e5)      # the whole point: many steps, so the log-time bunching shows

w_star = np.zeros(d)
w_star[:k] = 1
lambda_diag = np.arange(1, d + 1, dtype=float) ** (-2)
eta = 0.1

logistic_class_instance = LogisticRegressionClass(n, d, random_seed=85)

result = logistic_class_instance.LLM_generated_region_experiment_v1(
    number_of_trajectories=number_of_trajectories,
    w_star=w_star, d=d, n=n, t_steps=t_steps, eta=eta,
    use_lambda_diag=True, lambda_diag=lambda_diag,
    normalize_w_tilde=False,
    plot=False,          # we render our own interactive plot below
)
projected_trajectories = result["projected_trajectories"]
print("trajectories:", len(projected_trajectories), "| shape each:", projected_trajectories[0].shape)
'''

cell_plot_fn = '''\
TRAJ_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#8c564b", "#e377c2", "#17becf"]


def plot_trajectories_with_time(projected_trajectories, n_ticks=400, tick_every=None,
                                mark_points=None, w_tilde_dirs=None,
                                colorscale="Viridis", title=None,
                                width=820, height=820):
    """Interactive plotly view of projected GD trajectories with time encoded.

    - faint gray full path (Scattergl so 100k+ points stay responsive)
    - equal-step tick markers colored by log10(step); their DENSITY shows dwell time
    - hover on any tick -> exact step number
    - start = black square (step 0), end = red star (last step)

    n_ticks     : approx number of equal-step ticks per trajectory (ignored if tick_every set)
    tick_every  : put a tick every this many steps (overrides n_ticks)
    mark_points : optional dict {label: (x, y)} of extra reference points in the SAME
                  projected plane (e.g. {"w*": (||w*||, w*.w_2_dir)}), drawn as gold
                  diamonds so their position/magnitude can be compared to the trajectories.
    w_tilde_dirs: optional list of projected (x, y) for each trajectory's w_tilde
                  (max-margin direction). Drawn as a dashed ray from the origin through
                  that direction (color-matched per trajectory) plus a marker at the
                  actual w_tilde point, so you can see whether trajectory i is bending
                  toward its own w_tilde_i line.
    """
    fig = go.Figure()
    # shared color range so the colorbar means the same thing across trajectories
    T_max = max(proj.shape[0] for proj in projected_trajectories)
    cmin, cmax = 0.0, float(np.log10(T_max))
    # how far to extend the w_tilde rays: a bit past the farthest trajectory point
    ray_len = 1.15 * max(float(np.max(np.linalg.norm(proj, axis=1)))
                         for proj in projected_trajectories)

    for idx, proj in enumerate(projected_trajectories):
        T = proj.shape[0]
        stride = tick_every if tick_every is not None else max(1, T // n_ticks)
        tick_idx = np.arange(0, T, stride)
        cvals = np.log10(tick_idx + 1)

        # full faint path
        fig.add_trace(go.Scattergl(
            x=proj[:, 0], y=proj[:, 1], mode="lines",
            line=dict(width=1, color="rgba(150,150,150,0.5)"),
            name=f"traj {idx} path", legendgroup=f"traj{idx}",
            showlegend=False, hoverinfo="skip",
        ))
        # equal-step ticks, colored by log10(step)
        fig.add_trace(go.Scattergl(
            x=proj[tick_idx, 0], y=proj[tick_idx, 1], mode="markers",
            marker=dict(size=5, color=cvals, colorscale=colorscale,
                        cmin=cmin, cmax=cmax, showscale=False),
            text=[f"traj {idx}<br>step {int(t):,}" for t in tick_idx],
            hoverinfo="text", name=f"traj {idx}", legendgroup=f"traj{idx}",
        ))
        # start / end
        fig.add_trace(go.Scattergl(
            x=[proj[0, 0]], y=[proj[0, 1]], mode="markers",
            marker=dict(size=11, color="black", symbol="square"),
            text=["start (step 0)"], hoverinfo="text",
            legendgroup=f"traj{idx}", showlegend=False,
        ))
        fig.add_trace(go.Scattergl(
            x=[proj[-1, 0]], y=[proj[-1, 1]], mode="markers",
            marker=dict(size=13, color="red", symbol="star"),
            text=[f"end (step {T - 1:,})"], hoverinfo="text",
            legendgroup=f"traj{idx}", showlegend=False,
        ))

    # optional w_tilde_i directions: dashed ray from origin + marker, per trajectory
    if w_tilde_dirs is not None:
        for idx, (px, py) in enumerate(w_tilde_dirs):
            color = TRAJ_COLORS[idx % len(TRAJ_COLORS)]
            norm = float(np.hypot(px, py))
            if norm == 0:
                continue
            ux, uy = px / norm, py / norm
            # asymptote line (the direction the trajectory should converge to)
            fig.add_trace(go.Scattergl(
                x=[0.0, ray_len * ux], y=[0.0, ray_len * uy], mode="lines",
                line=dict(width=1.5, color=color, dash="dash"),
                name=f"w~_{idx} dir", legendgroup=f"wtilde{idx}",
                hoverinfo="skip",
            ))
            # marker at the actual w_tilde_i point
            fig.add_trace(go.Scattergl(
                x=[px], y=[py], mode="markers",
                marker=dict(size=9, color=color, symbol="x",
                            line=dict(width=1, color="black")),
                text=[f"w~_{idx}<br>({px:.3g}, {py:.3g})"], hoverinfo="text",
                name=f"w~_{idx}", legendgroup=f"wtilde{idx}", showlegend=False,
            ))

    # optional reference points (e.g. w*) in the same projected plane
    if mark_points:
        for label, (px, py) in mark_points.items():
            fig.add_trace(go.Scattergl(
                x=[px], y=[py], mode="markers+text",
                marker=dict(size=14, color="gold", symbol="diamond",
                            line=dict(width=1.5, color="black")),
                text=[label], textposition="top center",
                hovertext=[f"{label}<br>({px:.3g}, {py:.3g})"], hoverinfo="text",
                name=label, showlegend=True,
            ))

    fig.update_layout(
        title=title or "GD trajectories in (w*, sum w_tilde) plane — tick density & color = time",
        xaxis_title="w_1 = w* direction", yaxis_title="w_2 = sum(w_tilde) direction",
        width=width, height=height, template="plotly_white",
        legend=dict(title="trajectory"),
    )
    return fig
'''

cell_render = '''\
# project w* into the SAME plane so its size can be compared to the trajectories.
# axis 1 is the w* direction, so w*'s x-coordinate is exactly ||w*||.
w1, w2 = result["w_1_direction"], result["w_2_direction"]
w_star_proj = (float(w_star @ w1), float(w_star @ w2))

# project each trajectory's w_tilde_i (max-margin direction) into the plane, so we can
# see whether trajectory i is converging toward its own w~_i ray.
w_tilde_dirs = [(float(wt @ w1), float(wt @ w2)) for wt in result["w_tildes"]]

fig = plot_trajectories_with_time(
    projected_trajectories, n_ticks=400,
    mark_points={"w*": w_star_proj},
    w_tilde_dirs=w_tilde_dirs,
)
fig.show()
'''

cell_md_zoom = '''\
### Reading it
- Where ticks pile into a tight cluster, GD spent a huge number of steps moving very little — that\\'s the slow, late-time regime.
- Drag-select a rectangle on the plot to **zoom** into that cluster; hover any marker to read its step number and watch the late steps fan apart.
- Double-click to reset the zoom.

Tune `n_ticks` (or pass `tick_every=`) to trade off clutter vs. resolution.
'''

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_code_cell(cell_imports),
    nbf.v4.new_markdown_cell(cell_md_intro),
    nbf.v4.new_code_cell(cell_params),
    nbf.v4.new_code_cell(cell_plot_fn),
    nbf.v4.new_code_cell(cell_render),
    nbf.v4.new_markdown_cell(cell_md_zoom),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
