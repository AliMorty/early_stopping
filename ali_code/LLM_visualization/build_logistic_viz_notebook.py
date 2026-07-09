"""One-off generator for logistic_visualization.ipynb (interactive dashboard + pkl cache)."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "logistic_visualization.ipynb"

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

cell_cache = '''\
# --- setup-hashed cache with resume/extend ---
# Key on the SETUP only (not steps): running more steps for the same setup EXTENDS
# the cached trajectory instead of recomputing. GD is deterministic, so resuming from
# the stored last iterate on the stored data reproduces a from-scratch run exactly.
CACHE_DIR = "logistic_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SETUP_KEYS = ["n", "d", "k", "eta", "seed", "valid_size", "test_size"]
TRAJ_KEYS = ["log_loss_train", "log_loss_valid", "log_loss_test",
             "zero_one_train", "zero_one_valid", "zero_one_test"]

def config_key(setup):
    return hashlib.md5(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:8]

# lightweight index of past runs (setup + step count), so the dropdown never loads full pkls
INDEX_FILE = os.path.join(CACHE_DIR, "index.json")

def load_index():
    try:
        with open(INDEX_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _update_index(key, setup, steps):
    idx = load_index()
    idx[key] = {"setup": setup, "steps": steps}
    with open(INDEX_FILE, "w") as f:
        json.dump(idx, f, indent=2)

def _make_instance_and_data(setup):
    n, d, k = setup["n"], setup["d"], setup["k"]
    inst = LogisticRegressionClass(n, d, random_seed=setup["seed"], eta=setup["eta"])
    w_star = np.zeros(d); w_star[:k] = 1.0
    lam = np.arange(1, d + 1, dtype=float) ** (-2)
    data = {}
    data["X_train"], data["y_train"] = inst.generate_data(d, n,                 w_star, lambda_diag=lam)
    data["X_valid"], data["y_valid"] = inst.generate_data(d, setup["valid_size"], w_star, lambda_diag=lam)
    data["X_test"],  data["y_test"]  = inst.generate_data(d, setup["test_size"],  w_star, lambda_diag=lam)
    return inst, data, w_star

def _run_gd(inst, data, w_0, t_steps, eta):
    return inst.run_GD_for_t_steps(
        data["X_train"], data["y_train"], data["X_valid"], data["y_valid"],
        data["X_test"], data["y_test"],
        w_0=w_0, t_steps=t_steps, eta=eta,
        store_log_loss_traj_for_training=True,
        store_log_loss_traj_for_validation=True,
        store_log_loss_traj_for_test=True,
        store_zero_one_loss_traj_for_training=True,
        store_zero_one_loss_traj_for_validation=True,
        store_zero_one_loss_traj_for_test=True,
    )

def compute_and_cache(config):
    """Return (trajectories_dict, key, note, oracle_threshold). Trajectories always contain all
    6 curves, each of length exactly config['steps']. Repeated runs with more steps extend the
    cache. oracle_threshold is L_hat(w*) = training log-loss at w* (for the oracle stop line)."""
    setup = {k: config[k] for k in SETUP_KEYS}
    requested = config["steps"]
    key = config_key(setup)
    path = os.path.join(CACHE_DIR, f"logistic_{key}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        cached = rec["steps"]
        if cached >= requested:
            traj = {k: rec["trajectories"][k][:requested] for k in TRAJ_KEYS}
            return traj, key, f"cache (have {cached}, showing {requested})", rec.get("oracle_threshold")
        # extend: resume from stored last iterate on stored data
        inst = LogisticRegressionClass(setup["n"], setup["d"],
                                       random_seed=setup["seed"], eta=setup["eta"])
        delta = requested - cached
        new = _run_gd(inst, rec["data"], w_0=rec["w_last_iterate"], t_steps=delta, eta=setup["eta"])
        for k in TRAJ_KEYS:
            rec["trajectories"][k].extend(new[k])
        rec["steps"] = requested
        rec["w_last_iterate"] = new["w_last_iterate"]
        with open(path, "wb") as f:
            pickle.dump(rec, f)
        _update_index(key, setup, requested)
        return rec["trajectories"], key, f"extended (+{delta} -> {requested})", rec.get("oracle_threshold")

    # fresh run
    inst, data, w_star = _make_instance_and_data(setup)
    oracle_threshold = float(inst.oracular_ell_hat_w_star_zero_to_k(
        data["X_train"], data["y_train"], w_star))
    traj = _run_gd(inst, data, w_0=None, t_steps=requested, eta=setup["eta"])
    rec = {
        "setup": setup,
        "steps": requested,
        "trajectories": {k: list(traj[k]) for k in TRAJ_KEYS},
        "data": data,
        "w_last_iterate": traj["w_last_iterate"],
        "oracle_threshold": oracle_threshold,
    }
    with open(path, "wb") as f:
        pickle.dump(rec, f)
    _update_index(key, setup, requested)
    return rec["trajectories"], key, f"computed ({requested} steps)", oracle_threshold
'''

cell_dashboard = '''\
# --- two figures: log-loss and zero-one, each with train/valid/test traces ---
def make_fig(title, ytitle):
    f = go.FigureWidget()
    f.add_scatter(x=[], y=[], name="train", line=dict(color="steelblue"))
    f.add_scatter(x=[], y=[], name="valid", line=dict(color="tomato"))
    f.add_scatter(x=[], y=[], name="test",  line=dict(color="seagreen"))
    # stop lines as real traces so they appear in the legend and toggle by clicking it
    f.add_scatter(x=[], y=[], name="oracle stop", mode="lines",
                  line=dict(color="green",  dash="dash", width=1.5))
    f.add_scatter(x=[], y=[], name="val argmin", mode="lines",
                  line=dict(color="purple", dash="dash", width=1.5))
    f.update_layout(title=title, xaxis_title="step t", yaxis_title=ytitle, height=380,
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return f

fig_log = make_fig("Log loss",  "log loss")
fig_zo  = make_fig("Zero-one",  "0/1 error")

# --- flexible int parser: accepts "1e4", "1.5e3", "10k", "10000" ---
def parse_int(s):
    s = str(s).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(float(s))

# --- remember last-run parameters across sessions ---
LAST_PARAMS_FILE = "last_params.json"
_defaults = {"n": 100, "d": 200, "k": 10, "eta": 0.1, "seed": 85,
             "steps": 2000, "valid_size": 1000, "test_size": 1000}
def _load_last_params():
    try:
        with open(LAST_PARAMS_FILE) as f:
            return {**_defaults, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_defaults)
_p = _load_last_params()

# --- parameter inputs (initialized from last run) ---
st = {"description_width": "70px"}; lo = widgets.Layout(width="150px")
def T(desc, val): return widgets.Text(value=str(val), description=desc, style=st, layout=lo)
n_in     = T("n:", _p["n"]);     d_in    = T("d:", _p["d"]);       k_in    = T("k:", _p["k"])
eta_in   = T("eta:", _p["eta"]); seed_in = T("seed:", _p["seed"]); steps_in= T("steps:", _p["steps"])
valid_in = T("valid_size:", _p["valid_size"]); test_in = T("test_size:", _p["test_size"])

# (curves and stop lines are toggled by clicking legend entries — no checkboxes needed)

# --- X/Y linear|log scale toggle pairs (green = selected), styled like sonnet_visualization ---
def make_toggle_pair(label):
    b_lin = widgets.ToggleButton(value=True,  description="linear", button_style="success",
                                 layout=widgets.Layout(width="70px", height="28px"))
    b_log = widgets.ToggleButton(value=False, description="log",    button_style="",
                                 layout=widgets.Layout(width="70px", height="28px"))
    def on_lin(change):
        if change["new"]:
            b_log.value = False; b_log.button_style = ""; b_lin.button_style = "success"
        elif not b_log.value:
            b_lin.value = True
        update_scales()
    def on_log(change):
        if change["new"]:
            b_lin.value = False; b_lin.button_style = ""; b_log.button_style = "success"
        elif not b_lin.value:
            b_log.value = True
        update_scales()
    b_lin.observe(on_lin, names="value")
    b_log.observe(on_log, names="value")
    row = widgets.HBox([widgets.Label(label, layout=widgets.Layout(width="55px")), b_lin, b_log])
    return row, b_lin, b_log

x_row, x_lin, x_log = make_toggle_pair("X scale:")
y_row, y_lin, y_log = make_toggle_pair("Y scale:")

def update_scales():
    xt = "log" if x_log.value else "linear"
    yt = "log" if y_log.value else "linear"
    fig_log.update_layout(xaxis_type=xt, yaxis_type=yt)
    fig_zo.update_layout(xaxis_type=xt, yaxis_type=yt)

run_btn = widgets.Button(description="Run", button_style="primary", layout=widgets.Layout(width="120px"))
status  = widgets.Label(value="")

state = {"traj": None, "oracle_thr": None}

def _yrange(curves):
    vals = [v for c in curves for v in c]
    return (min(vals), max(vals)) if vals else (0.0, 1.0)

def _set_vline(trace, x, ymin, ymax):
    if x is None:
        trace.x, trace.y = [], []
    else:
        trace.x, trace.y = [x, x], [ymin, ymax]

def redraw():
    traj = state["traj"]
    if traj is None:
        return
    ts = list(range(len(traj["log_loss_train"])))
    thr = state["oracle_thr"]
    tr, va = traj["log_loss_train"], traj["log_loss_valid"]
    oi = next((t for t, l in enumerate(tr) if thr is not None and l <= thr), None)
    vi = 1 + int(np.argmin(va[1:])) if len(va) > 1 else None  # skip t=0 (w=0)

    log_curves = [traj["log_loss_train"], traj["log_loss_valid"], traj["log_loss_test"]]
    ly0, ly1 = _yrange(log_curves)
    with fig_log.batch_update():
        fig_log.data[0].x = ts; fig_log.data[0].y = traj["log_loss_train"]
        fig_log.data[1].x = ts; fig_log.data[1].y = traj["log_loss_valid"]
        fig_log.data[2].x = ts; fig_log.data[2].y = traj["log_loss_test"]
        _set_vline(fig_log.data[3], oi, ly0, ly1)
        _set_vline(fig_log.data[4], vi, ly0, ly1)

    zo_curves = [traj["zero_one_train"], traj["zero_one_valid"], traj["zero_one_test"]]
    zy0, zy1 = _yrange(zo_curves)
    with fig_zo.batch_update():
        fig_zo.data[0].x = ts; fig_zo.data[0].y = traj["zero_one_train"]
        fig_zo.data[1].x = ts; fig_zo.data[1].y = traj["zero_one_valid"]
        fig_zo.data[2].x = ts; fig_zo.data[2].y = traj["zero_one_test"]
        _set_vline(fig_zo.data[3], oi, zy0, zy1)
        _set_vline(fig_zo.data[4], vi, zy0, zy1)

def on_run(b):
    run_btn.disabled = True
    try:
        config = {
            "n": parse_int(n_in.value), "d": parse_int(d_in.value), "k": parse_int(k_in.value),
            "eta": float(eta_in.value), "seed": parse_int(seed_in.value),
            "steps": parse_int(steps_in.value),
            "valid_size": parse_int(valid_in.value), "test_size": parse_int(test_in.value),
        }
        with open(LAST_PARAMS_FILE, "w") as f:   # remember these for next session
            json.dump(config, f, indent=2)
        # peek the index to tell the user whether this run loads, extends, or computes fresh
        meta = load_index().get(config_key({k: config[k] for k in SETUP_KEYS}))
        req = config["steps"]
        if meta is None:
            status.value = f"running: computing {req} steps..."
        elif meta["steps"] >= req:
            status.value = "loading from memory..."
        else:
            status.value = f"loading from memory, then running +{req - meta['steps']} steps..."
        traj, key, note, oracle_thr = compute_and_cache(config)
        state["traj"] = traj
        state["oracle_thr"] = oracle_thr
        redraw()
        status.value = f"{note}  (key={key})"
        refresh_runs(select_key=key)  # keep dropdown current, select this run
    except Exception as e:
        status.value = f"Error: {e}"
    run_btn.disabled = False

run_btn.on_click(on_run)

# --- dropdown of past runs: select to load its params, then bump steps + Run to extend ---
run_select = widgets.Dropdown(options=[("- select a past run -", None)], value=None,
                              description="Past runs:", style={"description_width": "70px"},
                              layout=widgets.Layout(width="560px"))
refresh_btn = widgets.Button(description="refresh", icon="refresh",
                             layout=widgets.Layout(width="100px"))

def _fmt_run(key, meta):
    s = meta["setup"]
    return (f"n{s['n']} d{s['d']} k{s['k']} eta{s['eta']} seed{s['seed']} "
            f"v{s['valid_size']} t{s['test_size']}  |  {meta['steps']} steps  [{key}]")

def refresh_runs(select_key=None):
    idx = load_index()
    opts = [("- select a past run -", None)] + [(_fmt_run(k, idx[k]), k) for k in sorted(idx)]
    run_select.options = opts
    run_select.value = select_key if any(select_key == k for _, k in opts) else None

def on_select(change):
    key = change["new"]
    if not key:
        return
    meta = load_index().get(key)
    if not meta:
        return
    s = meta["setup"]
    n_in.value = str(s["n"]); d_in.value = str(s["d"]); k_in.value = str(s["k"])
    eta_in.value = str(s["eta"]); seed_in.value = str(s["seed"])
    valid_in.value = str(s["valid_size"]); test_in.value = str(s["test_size"])
    steps_in.value = str(meta["steps"])

run_select.observe(on_select, names="value")
refresh_btn.on_click(lambda b: refresh_runs())
refresh_runs()  # populate on open

display(widgets.VBox([
    widgets.HBox([x_row, y_row]),
    widgets.HBox([run_select, refresh_btn]),
    widgets.HBox([n_in, d_in, k_in, eta_in]),
    widgets.HBox([seed_in, steps_in, valid_in, test_in]),
    widgets.HBox([run_btn, status]),
    widgets.HTML('<hr style="margin:6px 0">'),
    widgets.HBox([fig_log, fig_zo], layout=widgets.Layout(flex_flow="row wrap")),
]))
'''

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell(cell_imports),
            nbf.v4.new_code_cell(cell_cache),
            nbf.v4.new_code_cell(cell_dashboard)]
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
