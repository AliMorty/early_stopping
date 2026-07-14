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
import os, sys, json, hashlib, pickle
# logistic_regression.py (Ali's nbconvert'd algorithm) lives in ali_code/, the parent of this LLM_visualization/ folder
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

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
# Only scalars here (cheap). The heavy GD runs happen in the plot cells below and are
# cached, so re-running an identical setup reloads instead of recomputing.
n = 100
d = 200
k = 10
number_of_trajectories = 3
t_steps = int(1e5)      # the whole point: many steps, so the log-time bunching shows
eta = 0.1
seed = 85               # class random_seed; part of a run's identity
'''

cell_plot_fn = '''\
TRAJ_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#8c564b", "#e377c2", "#17becf"]


def plot_trajectories_with_time(projected_trajectories, n_ticks=400, tick_every=None,
                                mark_points=None, w_tilde_dirs=None,
                                early_stop_pts=None, argmin_pts=None, title=None,
                                width=820, height=820):
    """Interactive plotly view of projected GD trajectories with time encoded.

    Each trajectory gets one color (matched to its w_tilde ray), so trajectory i
    is visually tied to its own asymptote w~_i.

    - full path (Scattergl so 100k+ points stay responsive)
    - equal-step tick markers; their DENSITY (not color) shows dwell time
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
    early_stop_pts: optional list of projected (x, y) of each trajectory's oracular
                  early-stop iterate (green square). One legend entry, toggle separately.
    argmin_pts  : optional list of projected (x, y) of each trajectory's population
                  test-loss argmin iterate (purple star). One legend entry, toggle separately.
    """
    fig = go.Figure()
    # how far to extend the w_tilde rays: a bit past the farthest trajectory point
    ray_len = 1.15 * max(float(np.max(np.linalg.norm(proj, axis=1)))
                         for proj in projected_trajectories)

    for idx, proj in enumerate(projected_trajectories):
        color = TRAJ_COLORS[idx % len(TRAJ_COLORS)]
        T = proj.shape[0]
        stride = tick_every if tick_every is not None else max(1, T // n_ticks)
        tick_idx = np.arange(0, T, stride)

        # full path, in this trajectory's color
        fig.add_trace(go.Scattergl(
            x=proj[:, 0], y=proj[:, 1], mode="lines",
            line=dict(width=1, color=color),
            name=f"traj {idx} path", legendgroup=f"traj{idx}",
            showlegend=False, hoverinfo="skip",
        ))
        # equal-step ticks: same color; DENSITY conveys time (dense = many steps here)
        fig.add_trace(go.Scattergl(
            x=proj[tick_idx, 0], y=proj[tick_idx, 1], mode="markers",
            marker=dict(size=5, color=color),
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

    # optional stopping points, one legend entry per type so they toggle separately
    def _stop_trace(pts, color, symbol, name, size):
        return go.Scattergl(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(width=1.5, color="black")),
            text=[f"{name} (traj {i})" for i in range(len(pts))],
            hoverinfo="text", name=name,
        )
    if early_stop_pts:
        fig.add_trace(_stop_trace(early_stop_pts, "green", "square", "early stop", 12))
    if argmin_pts:
        fig.add_trace(_stop_trace(argmin_pts, "purple", "star", "test-loss argmin", 15))

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

cell_cache = '''\
# --- run cache: save only the 2D-projected data needed to redraw a plot ---
# GD is the slow part; once a setup is run we pickle the compact plot data so an
# identical setup reloads instantly. index.json lets the dropdown (bottom of the
# notebook) browse past runs without opening any pkl.
CACHE_DIR = "region_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_FILE = os.path.join(CACHE_DIR, "index.json")

# fields stored per run (also the searchable label)
SETUP_KEYS = ["version", "n", "d", "k", "number_of_trajectories", "t_steps",
              "eta", "seed", "test_sample_size", "normalize_w_tilde", "use_lambda_diag"]
# a run's IDENTITY ignores t_steps: asking for more steps extends the same entry, so
# there is exactly one cache entry per setting and it holds the highest T computed.
IDENTITY_KEYS = [key for key in SETUP_KEYS if key != "t_steps"]

def config_key(setup):
    payload = {k: setup[k] for k in IDENTITY_KEYS}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]

def load_index():
    try:
        with open(INDEX_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _update_index(key, setup):
    idx = load_index()
    idx[key] = {k: setup[k] for k in SETUP_KEYS}
    with open(INDEX_FILE, "w") as f:
        json.dump(idx, f, indent=2)

def run_label(key, s):
    return (f"[{key}] {s['version']} | n{s['n']} d{s['d']} k{s['k']} "
            f"traj{s['number_of_trajectories']} T{s['t_steps']} eta{s['eta']} "
            f"seed{s['seed']} test{s['test_sample_size']}")

def run_or_load_region(setup, force=False):
    """Load compact plot data for `setup` from cache, or run the experiment + save it.
    setup['version'] selects v1 (no stop markers) or v2 (early-stop + test argmin).
    Returns (plotdata, key). Pass force=True to recompute and overwrite."""
    setup = dict(setup)
    key = config_key(setup)
    path = os.path.join(CACHE_DIR, f"region_{key}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            pd = pickle.load(f)
        # cached entry already has >= the requested steps: reuse it (it's the highest T).
        if pd["setup"]["t_steps"] >= setup["t_steps"]:
            print("loaded", run_label(key, pd["setup"]))
            return pd, key
        # asking for more steps: recompute at the larger T and overwrite the one entry.
        print(f"extending T {pd['setup']['t_steps']} -> {setup['t_steps']} (recompute)")

    w_star = np.zeros(setup["d"]); w_star[:setup["k"]] = 1.0
    lam = np.arange(1, setup["d"] + 1, dtype=float) ** (-2)
    inst = LogisticRegressionClass(setup["n"], setup["d"],
                                   random_seed=setup["seed"], eta=setup["eta"])
    common = dict(number_of_trajectories=setup["number_of_trajectories"], w_star=w_star,
                  d=setup["d"], n=setup["n"], t_steps=setup["t_steps"], eta=setup["eta"],
                  use_lambda_diag=setup["use_lambda_diag"], lambda_diag=lam,
                  normalize_w_tilde=setup["normalize_w_tilde"], plot=False)
    if setup["version"] == "v2":
        result = inst.LLM_generated_region_experiment_v2(
            test_sample_size=setup["test_sample_size"],
            measure_population_loss_for_iterates=True, **common)
        early_stop_t = list(result["early_stop_t"])
        test_loss_argmin_t = list(result["test_loss_argmin_t"])
    else:
        result = inst.LLM_generated_region_experiment_v1(**common)
        early_stop_t = [None] * setup["number_of_trajectories"]
        test_loss_argmin_t = [None] * setup["number_of_trajectories"]

    w1, w2 = result["w_1_direction"], result["w_2_direction"]
    pd = {
        "setup": {k: setup[k] for k in SETUP_KEYS},
        "projected_trajectories": [np.asarray(p) for p in result["projected_trajectories"]],
        "w_star_proj": (float(w_star @ w1), float(w_star @ w2)),
        "w_tilde_dirs": [(float(wt @ w1), float(wt @ w2)) for wt in result["w_tildes"]],
        "early_stop_t": early_stop_t,
        "test_loss_argmin_t": test_loss_argmin_t,
    }
    with open(path, "wb") as f:
        pickle.dump(pd, f)
    _update_index(key, setup)
    print("computed + saved", run_label(key, pd["setup"]))
    return pd, key

def load_region_run(key):
    with open(os.path.join(CACHE_DIR, f"region_{key}.pkl"), "rb") as f:
        return pickle.load(f)

def plot_from_plotdata(pd, n_ticks=400, title=None):
    """Rebuild the interactive figure from cached compact plot data."""
    proj = pd["projected_trajectories"]
    es = [tuple(proj[i][t]) for i, t in enumerate(pd["early_stop_t"]) if t is not None]
    am = [tuple(proj[i][t]) for i, t in enumerate(pd["test_loss_argmin_t"]) if t is not None]
    return plot_trajectories_with_time(
        proj, n_ticks=n_ticks, mark_points={"w*": pd["w_star_proj"]},
        w_tilde_dirs=pd["w_tilde_dirs"], early_stop_pts=es, argmin_pts=am,
        title=title or f"region run [{config_key(pd['setup'])}]",
    )
'''

cell_render = '''\
# V1 plot (no stop markers). Cached: identical setup reloads instead of re-running GD.
setup_v1 = dict(version="v1", n=n, d=d, k=k,
                number_of_trajectories=number_of_trajectories, t_steps=t_steps,
                eta=eta, seed=seed, test_sample_size=None,
                normalize_w_tilde=False, use_lambda_diag=True)
pd1, key1 = run_or_load_region(setup_v1)          # force=True to recompute
plot_from_plotdata(pd1, n_ticks=400,
                   title=f"V1 region run [{key1}]").show()
'''

cell_md_zoom = '''\
### Reading it
- Where ticks pile into a tight cluster, GD spent a huge number of steps moving very little — that\\'s the slow, late-time regime.
- Drag-select a rectangle on the plot to **zoom** into that cluster; hover any marker to read its step number and watch the late steps fan apart.
- Double-click to reset the zoom.

Tune `n_ticks` (or pass `tick_every=`) to trade off clutter vs. resolution.
'''

cell_md_v2 = '''\
## Version 2 — with early-stopping & test-loss-argmin markers

`LLM_generated_region_experiment_v2` additionally generates a large held-out test set
(`test_sample_size`, ~population loss) and, per trajectory, records two iterates:

- **early stop** (green square) — first `t` with training loss `<= L_hat(w*_{0:k})`
  (the oracular early-stopping rule from `plot_loss_over_time`).
- **test-loss argmin** (purple star) — `argmin_t` of the population (test) logistic loss,
  i.e. the best iterate to have stopped at.

Both are indices into the trajectory (`loss[t]` lines up with `w_trajectory[t]`), so they
drop directly onto the projected path. Toggle each via its legend entry to compare where,
in the plane, the practical stop lands vs. the optimum. They usually sit deep in the
end-cluster — zoom in.
'''

cell_render_v2 = '''\
# V2 plot: early-stop (green square) + test-loss argmin (purple star). Cached.
setup_v2 = dict(version="v2", n=n, d=d, k=k,
                number_of_trajectories=number_of_trajectories, t_steps=t_steps,
                eta=eta, seed=seed, test_sample_size=int(3e3),
                normalize_w_tilde=False, use_lambda_diag=True)
pd2, key2 = run_or_load_region(setup_v2)          # force=True to recompute
plot_from_plotdata(pd2, n_ticks=400,
                   title=f"V2: early-stop (green sq) & test-loss argmin (purple star) [{key2}]").show()
'''

cell_md_panel = '''\
## Control panel

Enter a setup, pick **v1** (no stop markers) or **v2** (early-stop green square +
test-loss argmin purple star), and hit **Run / load** — it computes GD once and caches
it, or reloads instantly if that exact setup was run before. The **saved runs** dropdown
lists every cached run (selecting one refills the fields and redraws it); the `n_ticks`
slider re-renders the current run at a different tick density. `t_steps` accepts `1e5` /
`100k` / `100000`.

There is **one cache entry per setting** (everything except `t_steps`). Asking for a
larger `t_steps` on the same setting extends that single entry to the higher T; asking
for a smaller/equal `t_steps` just reloads the stored (highest-T) run.
'''

cell_panel = '''\
def _parse_int(s):
    s = str(s).strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(float(s))

_st = {"description_width": "115px"}; _lo = widgets.Layout(width="215px")
def _T(desc, val): return widgets.Text(value=str(val), description=desc, style=_st, layout=_lo)

w_ver   = widgets.Dropdown(options=["v2", "v1"], value="v2", description="version:",
                           style=_st, layout=_lo)
w_n     = _T("n:", n);      w_d = _T("d:", d);   w_k = _T("k:", k)
w_ntraj = _T("n_traj:", number_of_trajectories); w_tstep = _T("t_steps:", t_steps)
w_eta   = _T("eta:", eta);  w_seed = _T("seed:", seed)
w_test  = _T("test_size:", int(3e3))
w_norm  = widgets.Checkbox(value=False, description="normalize_w_tilde", indent=False)
w_ticks = widgets.IntSlider(value=400, min=50, max=1500, step=50, description="n_ticks:",
                            style=_st, layout=widgets.Layout(width="340px"))

run_btn = widgets.Button(description="Run / load", button_style="primary",
                         layout=widgets.Layout(width="120px"))
run_dd  = widgets.Dropdown(options=[], description="saved runs:", style=_st,
                           layout=widgets.Layout(width="680px"))
status  = widgets.Label(value="")
out     = widgets.Output()

def _setup_from_inputs():
    ver = w_ver.value
    return dict(version=ver, n=_parse_int(w_n.value), d=_parse_int(w_d.value),
                k=_parse_int(w_k.value), number_of_trajectories=_parse_int(w_ntraj.value),
                t_steps=_parse_int(w_tstep.value), eta=float(w_eta.value),
                seed=_parse_int(w_seed.value),
                test_sample_size=(_parse_int(w_test.value) if ver == "v2" else None),
                normalize_w_tilde=bool(w_norm.value), use_lambda_diag=True)

def _fill_inputs(s):
    w_ver.value = s["version"]; w_n.value = str(s["n"]); w_d.value = str(s["d"])
    w_k.value = str(s["k"]); w_ntraj.value = str(s["number_of_trajectories"])
    w_tstep.value = str(s["t_steps"]); w_eta.value = str(s["eta"])
    w_seed.value = str(s["seed"])
    if s["test_sample_size"] is not None: w_test.value = str(s["test_sample_size"])
    w_norm.value = bool(s["normalize_w_tilde"])

def _refresh_dd():
    run_dd.unobserve(_on_select, names="value")
    idx = load_index()
    run_dd.options = [(run_label(k, s), k) for k, s in sorted(idx.items())]
    run_dd.observe(_on_select, names="value")

def _plot_key(key):
    with out:
        out.clear_output(wait=True)
        plot_from_plotdata(load_region_run(key), n_ticks=w_ticks.value).show()

def _on_run(_=None):
    status.value = "running / loading (GD can take a while the first time)..."
    try:
        pd, key = run_or_load_region(_setup_from_inputs())
    except Exception as e:
        status.value = f"error: {e}"; raise
    _refresh_dd()
    if run_dd.value == key:
        _plot_key(key)          # value unchanged -> observer won't fire; plot once here
    else:
        run_dd.value = key      # value change fires _on_select -> plots once
    status.value = f"showing [{key}]"

def _on_select(change):
    key = change["new"]
    if key is None: return
    _fill_inputs(load_region_run(key)["setup"]); _plot_key(key)

def _on_ticks(_):
    if run_dd.value is not None: _plot_key(run_dd.value)

run_btn.on_click(_on_run)
run_dd.observe(_on_select, names="value")
w_ticks.observe(_on_ticks, names="value")

display(widgets.VBox([
    widgets.HBox([w_ver, w_n, w_d, w_k]),
    widgets.HBox([w_ntraj, w_tstep, w_eta, w_seed]),
    widgets.HBox([w_test, w_norm, w_ticks]),
    widgets.HBox([run_btn, status]),
    run_dd,
    out,
]))
_refresh_dd()
with out:
    print("enter a setup and click Run / load, or pick a saved run from the dropdown.")
'''

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_code_cell(cell_imports),
    nbf.v4.new_markdown_cell(cell_md_intro),
    nbf.v4.new_code_cell(cell_params),      # default values that prefill the control panel
    nbf.v4.new_code_cell(cell_plot_fn),
    nbf.v4.new_code_cell(cell_cache),
    nbf.v4.new_markdown_cell(cell_md_v2),
    nbf.v4.new_markdown_cell(cell_md_zoom),
    nbf.v4.new_markdown_cell(cell_md_panel),
    nbf.v4.new_code_cell(cell_panel),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
