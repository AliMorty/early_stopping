"""One-off generator for wtilde_average_convergence.ipynb.

Experiment: treat each max-margin solution w_tilde as one draw from a random
distribution (randomness = a fresh (X, y) sample from generate_data). Draw it N
times, accumulate the running average  wbar_N = (1/N) sum_n w_tilde_n, and watch
what wbar_N (and its population loss) converge to. Question: does wbar_N -> w*
and L(wbar_N) -> L(w*)?

For each N (x-axis = number of draws) we plot:
  - ||wbar_N - w*||         (norm difference)
  - angle(wbar_N, w*) [deg] (direction only; scale-invariant)
  - L(wbar_N)               (population logistic loss, evaluated on a fresh sample)
  - L(w*)                   (constant reference line)

Two toggles matter:
  - normalize_w_tilde : if True, EACH w_tilde is normalized to unit norm before
    averaging, so the average focuses purely on direction. w_tilde itself is the
    UNnormalized max-margin solution by default (as requested).
  - the loss L is always measured on a FRESH, large held-out sample (population
    loss estimate), never on any training set (there is no training here).

The figure mirrors region_visualization: one interactive plotly figure whose
traces are toggled by clicking legend entries, plus a control panel + run cache.

Regenerate: python ali_code/LLM_visualization/build_wtilde_average_convergence_notebook.py
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "wtilde_average_convergence.ipynb"

cell_rebuild = '''\
# Rebuild logistic_regression.py from Ali's notebook so we import the current class.
# (Regenerating a derived artifact is allowed; the notebook stays authoritative.)
import subprocess, sys, os
print(subprocess.run([sys.executable, "build_logistic_module.py"],
                     capture_output=True, text=True).stdout)
'''

cell_imports = '''\
import os, sys, json, hashlib, pickle
# logistic_regression.py (Ali's nbconvert'd algorithm) lives in ali_code/, the
# parent of this LLM_visualization/ folder.
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

from logistic_regression import LogisticRegressionClass
'''

cell_md_intro = '''\
# Does the average of many `w_tilde` converge to `w*`?

Each max-margin solution `w_tilde = find_w_tilde(X, y)` is one **draw from a random
distribution** — the randomness is a fresh `(X, y)` sample from `generate_data`.
Draw it `N` times and form the running average

$$\\bar w_N = \\frac{1}{N}\\sum_{n=1}^{N} \\tilde w_n .$$

We ask whether `wbar_N -> w*` and `L(wbar_N) -> L(w*)`. For each `N` (x-axis =
number of draws) we plot:

- **`||wbar_N - w*||`** — norm difference,
- **`angle(wbar_N, w*)`** [degrees] — direction only (scale-invariant),
- **`L(wbar_N)`** — population logistic loss, measured on a **fresh large sample**,
- **`L(w*)`** — constant reference line.

Notes:
- `w_tilde` is the **unnormalized** max-margin solution by default. Its norm is
  much larger than `||w*||`, so `||wbar_N - w*||` need not go to 0 — the **angle**
  is the meaningful convergence signal there.
- Tick the **`normalize_w_tilde`** box to normalize *each* `w_tilde` to unit norm
  before averaging, focusing the average purely on direction.
- `L` is always a **population** estimate on a fresh held-out sample — there is no
  training set involved.

Traces are toggled by **clicking legend entries** (as in `region_visualization`).
'''

cell_params = '''\
# --- experiment setup (these values prefill the control panel below) ---
n = 100          # rows per (X, y) draw used to solve for one w_tilde
d = 200          # dimension
k = 10           # w* = first k coords are 1, rest 0
N_draws = 300    # how many w_tilde to draw (x-axis goes 1..N_draws)
eta = 0.1        # class param (unused by this experiment, kept for identity)
seed = 85        # seeds the draw RNG; part of a run's identity
test_size = 3000 # size of the fresh sample used to estimate the population loss L
'''

cell_experiment = '''\
def _angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    c = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def run_wtilde_average(setup):
    """Draw N w_tilde, accumulate the running mean, and record convergence metrics.

    Returns a dict of 1-D arrays indexed by N = 1..N_draws:
        Ns, norm_diff, angle_deg, loss_curve   (+ scalar L_w_star, and wbar_final).
    Each draw uses a FRESH (X, y) from generate_data (independent, since generate_data
    advances the class RNG). L is estimated on ONE fresh held-out sample of size
    test_size drawn from a separate RNG so it does not perturb the draw sequence.
    """
    d, n, k = setup["d"], setup["n"], setup["k"]
    N_draws = setup["N_draws"]
    w_star = np.zeros(d); w_star[:k] = 1.0
    lam = np.arange(1, d + 1, dtype=float) ** (-2)

    # draws use this instance's RNG (seeded, reproducible)
    inst = LogisticRegressionClass(n, d, random_seed=setup["seed"], eta=setup["eta"])
    # fresh, independent population sample for evaluating L (different seed)
    inst_pop = LogisticRegressionClass(setup["test_size"], d,
                                       random_seed=setup["seed"] + 9973, eta=setup["eta"])
    Xp, yp = inst_pop.generate_data(d, setup["test_size"], w_star,
                                    lambda_diag=lam, use_lambda_diag=True)
    L_w_star = float(inst_pop.logistic_loss(w_star, Xp, yp))

    running_sum = np.zeros(d)
    norm_diff = np.empty(N_draws)
    angle_deg = np.empty(N_draws)
    loss_curve = np.empty(N_draws)

    for i in range(N_draws):
        X, y = inst.generate_data(d, n, w_star, lambda_diag=lam, use_lambda_diag=True)
        w_tilde = inst.find_w_tilde(X, y, normalize=setup["normalize_w_tilde"])
        running_sum += w_tilde
        wbar = running_sum / (i + 1)
        norm_diff[i] = float(np.linalg.norm(wbar - w_star))
        angle_deg[i] = _angle_deg(wbar, w_star)
        loss_curve[i] = float(inst_pop.logistic_loss(wbar, Xp, yp))

    return {
        "Ns": np.arange(1, N_draws + 1),
        "norm_diff": norm_diff,
        "angle_deg": angle_deg,
        "loss_curve": loss_curve,
        "L_w_star": L_w_star,
        "wbar_final": running_sum / N_draws,
    }
'''

cell_cache = '''\
# --- run cache: identical setups reload instead of re-solving all the QPs ---
CACHE_DIR = "wtilde_avg_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_FILE = os.path.join(CACHE_DIR, "index.json")

SETUP_KEYS = ["n", "d", "k", "N_draws", "eta", "seed", "test_size", "normalize_w_tilde"]
# a run's IDENTITY ignores N_draws: asking for more draws extends the same entry, so
# there is exactly one cache entry per setting holding the highest N computed.
IDENTITY_KEYS = [key for key in SETUP_KEYS if key != "N_draws"]

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
    return (f"[{key}] n{s['n']} d{s['d']} k{s['k']} N{s['N_draws']} "
            f"seed{s['seed']} test{s['test_size']} "
            f"{'unit-w~' if s['normalize_w_tilde'] else 'raw-w~'}")

def _truncate(data, N):
    out = dict(data)
    for kk in ("Ns", "norm_diff", "angle_deg", "loss_curve"):
        out[kk] = data[kk][:N]
    return out

def run_or_load(setup, force=False):
    """Load cached metrics for `setup`, or run the experiment + save. Returns (data, key).
    N_draws is treated like a length: a cached run with >= N_draws is reused (truncated);
    a larger request recomputes and overwrites the single entry."""
    setup = dict(setup)
    key = config_key(setup)
    path = os.path.join(CACHE_DIR, f"wavg_{key}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob["setup"]["N_draws"] >= setup["N_draws"]:
            print("loaded", run_label(key, blob["setup"]))
            return _truncate(blob["data"], setup["N_draws"]), key
        print(f"extending N {blob['setup']['N_draws']} -> {setup['N_draws']} (recompute)")

    data = run_wtilde_average(setup)
    blob = {"setup": {k: setup[k] for k in SETUP_KEYS}, "data": data}
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    _update_index(key, setup)
    print("computed + saved", run_label(key, setup))
    return data, key

def load_run(key):
    with open(os.path.join(CACHE_DIR, f"wavg_{key}.pkl"), "rb") as f:
        return pickle.load(f)
'''

cell_plot_fn = '''\
def plot_convergence(data, setup, key="", log_x=True, log_y=False):
    """Interactive plotly line plot of the convergence metrics vs number of draws N.

    Every metric is its own legend-toggleable trace (click the legend to hide/show),
    exactly like region_visualization. The metrics live on different scales, so the
    log-y toggle and per-trace toggling let you focus on one at a time.
    """
    Ns = data["Ns"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ns, y=data["norm_diff"], mode="lines",
                             name="||wbar_N - w*||", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=Ns, y=data["angle_deg"], mode="lines",
                             name="angle(wbar_N, w*) [deg]", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=Ns, y=data["loss_curve"], mode="lines",
                             name="L(wbar_N)", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=[Ns[0], Ns[-1]], y=[data["L_w_star"], data["L_w_star"]],
                             mode="lines", name="L(w*)",
                             line=dict(color="#2ca02c", dash="dash", width=1)))
    norm_txt = "unit-normalized w~" if setup["normalize_w_tilde"] else "raw (unnormalized) w~"
    fig.update_layout(
        title=(f"average of w~ vs number of draws [{key}] — {norm_txt}<br>"
               f"<sub>n={setup['n']} d={setup['d']} k={setup['k']} "
               f"seed={setup['seed']} test_size={setup['test_size']}</sub>"),
        xaxis_title="N = number of w~ draws",
        yaxis_title="metric value",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        width=900, height=620, template="plotly_white",
        legend=dict(title="click to toggle"),
    )
    return fig
'''

cell_md_panel = '''\
## Control panel

Set `n`, `d`, `k`, number of draws `N`, `seed`, the fresh-sample size `test_size`,
and the **`normalize_w_tilde`** toggle, then hit **Run / load**. It solves one
max-margin QP per draw (slow the first time), caches the result, and reloads
instantly for an identical setup. The **saved runs** dropdown browses past runs
(selecting one refills the fields and redraws). `log x` / `log y` re-render the
current run on log axes without recomputing.

There is **one cache entry per setting** (everything except `N`). Asking for a
larger `N` extends that entry; a smaller/equal `N` just reloads (and truncates)
the stored run.
'''

cell_panel = '''\
def _parse_int(s):
    s = str(s).strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(float(s))

_st = {"description_width": "110px"}; _lo = widgets.Layout(width="210px")
def _T(desc, val): return widgets.Text(value=str(val), description=desc, style=_st, layout=_lo)

w_n     = _T("n:", n);      w_d = _T("d:", d);   w_k = _T("k:", k)
w_N     = _T("N draws:", N_draws)
w_eta   = _T("eta:", eta);  w_seed = _T("seed:", seed)
w_test  = _T("test_size:", test_size)
w_norm  = widgets.Checkbox(value=False, description="normalize_w_tilde", indent=False)
w_logx  = widgets.Checkbox(value=True, description="log x", indent=False)
w_logy  = widgets.Checkbox(value=False, description="log y", indent=False)

run_btn = widgets.Button(description="Run / load", button_style="primary",
                         layout=widgets.Layout(width="120px"))
run_dd  = widgets.Dropdown(options=[], description="saved runs:", style=_st,
                           layout=widgets.Layout(width="640px"))
status  = widgets.Label(value="")
out     = widgets.Output()

def _setup_from_inputs():
    return dict(n=_parse_int(w_n.value), d=_parse_int(w_d.value), k=_parse_int(w_k.value),
                N_draws=_parse_int(w_N.value), eta=float(w_eta.value),
                seed=_parse_int(w_seed.value), test_size=_parse_int(w_test.value),
                normalize_w_tilde=bool(w_norm.value))

def _fill_inputs(s):
    w_n.value = str(s["n"]); w_d.value = str(s["d"]); w_k.value = str(s["k"])
    w_N.value = str(s["N_draws"]); w_eta.value = str(s["eta"])
    w_seed.value = str(s["seed"]); w_test.value = str(s["test_size"])
    w_norm.value = bool(s["normalize_w_tilde"])

def _refresh_dd():
    run_dd.unobserve(_on_select, names="value")
    idx = load_index()
    run_dd.options = [(run_label(k, s), k) for k, s in sorted(idx.items())]
    run_dd.observe(_on_select, names="value")

def _plot_key(key):
    with out:
        out.clear_output(wait=True)
        blob = load_run(key)
        data = _truncate(blob["data"], _parse_int(w_N.value)) \\
            if blob["setup"]["N_draws"] >= _parse_int(w_N.value) else blob["data"]
        plot_convergence(data, blob["setup"], key=key,
                         log_x=w_logx.value, log_y=w_logy.value).show()

def _on_run(_=None):
    status.value = "running / loading (one QP per draw — can take a while the first time)..."
    try:
        data, key = run_or_load(_setup_from_inputs())
    except Exception as e:
        status.value = f"error: {e}"; raise
    _refresh_dd()
    if run_dd.value == key:
        _plot_key(key)
    else:
        run_dd.value = key
    status.value = f"showing [{key}]"

def _on_select(change):
    key = change["new"]
    if key is None: return
    _fill_inputs(load_run(key)["setup"]); _plot_key(key)

def _on_axis(_):
    if run_dd.value is not None: _plot_key(run_dd.value)

run_btn.on_click(_on_run)
run_dd.observe(_on_select, names="value")
w_logx.observe(_on_axis, names="value")
w_logy.observe(_on_axis, names="value")

display(widgets.VBox([
    widgets.HBox([w_n, w_d, w_k, w_N]),
    widgets.HBox([w_eta, w_seed, w_test]),
    widgets.HBox([w_norm, w_logx, w_logy]),
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
    nbf.v4.new_code_cell(cell_rebuild),
    nbf.v4.new_code_cell(cell_imports),
    nbf.v4.new_markdown_cell(cell_md_intro),
    nbf.v4.new_code_cell(cell_params),
    nbf.v4.new_code_cell(cell_experiment),
    nbf.v4.new_code_cell(cell_cache),
    nbf.v4.new_code_cell(cell_plot_fn),
    nbf.v4.new_markdown_cell(cell_md_panel),
    nbf.v4.new_code_cell(cell_panel),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
