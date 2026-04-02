"""Generate dashboard_2/dashboard.html with per-metric normalize toggles."""
import glob
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from plotting import load_run, compute_metrics

# ── Load data ────────────────────────────────────────────────────────────────
pkl_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "gd_trajectories/run_*.pkl")))
print(f"Found {len(pkl_files)} runs")

runs_by_ratio = {}
for f in pkl_files:
    data = load_run(f)
    metrics = compute_metrics(data)
    if not metrics:
        continue
    config = data["config"]
    n, d, k, seed = config["n"], config["d"], config["k"], config["seed"]
    ratio = round(k / n, 6)
    entry = {
        "label": f"k/n={ratio:.4f} (k={k},n={n},d={d},seed={seed})",
        "k": int(k), "n": int(n), "d": int(d), "seed": int(seed),
        "times": metrics["times"].tolist(),
    }
    if "norms"            in metrics: entry["norm"]          = metrics["norms"].tolist()
    if "loss_values"      in metrics: entry["train_loss"]    = metrics["loss_values"].tolist();  entry["loss_times"] = metrics["loss_times"].tolist()
    if "pop_loss_values"  in metrics: entry["pop_loss"]      = metrics["pop_loss_values"].tolist(); entry["pop_loss_times"] = metrics["pop_loss_times"].tolist()
    if "angle_w_star"     in metrics: entry["angle_w_star"]  = metrics["angle_w_star"].tolist()
    if "angle_w_tilde"    in metrics: entry["angle_w_tilde"] = metrics["angle_w_tilde"].tolist()
    if "stopping_times"   in metrics: entry["stopping_times"] = [int(t) for t in metrics["stopping_times"]]
    if "w_star_norm"      in metrics: entry["w_star_norm"]   = float(metrics["w_star_norm"])
    runs_by_ratio.setdefault(ratio, []).append(entry)

sorted_ratios = sorted(runs_by_ratio.keys())
all_runs = [run for r in sorted_ratios for run in runs_by_ratio[r]]
json_blob = json.dumps({"runs": all_runs})
print(f"Serialized {len(all_runs)} runs")

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GD Trajectory Dashboard 2</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }
  h1 { text-align: center; margin-bottom: 18px; font-size: 1.5em; color: #333; }
  .controls { background: #fff; border-radius: 8px; padding: 18px 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .slider-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .slider-row label { font-weight: 600; white-space: nowrap; }
  .slider-row input[type=range] { flex: 1; }
  .slider-row .slider-label { min-width: 340px; font-size: 0.95em; color: #555; }
  .step-btn { width: 32px; height: 32px; font-size: 1.1em; font-weight: 700; cursor: pointer; border: 1px solid #ccc; border-radius: 4px; background: #fff; color: #333; display: flex; align-items: center; justify-content: center; }
  .step-btn:hover { background: #eee; }
  .step-btn:active { background: #ddd; }
  .top-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; margin-bottom: 12px; }
  .top-row label { font-weight: 600; margin-right: 4px; }
  .ctrl-select { padding: 4px 8px; font-size: 0.95em; }

  /* metric table */
  .metric-table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
  .metric-table td { padding: 5px 10px; vertical-align: middle; border-bottom: 1px solid #f0f0f0; }
  .metric-table tr:last-child td { border-bottom: none; }
  .metric-name { display: flex; align-items: center; gap: 6px; font-size: 0.95em; min-width: 120px; }
  .metric-name input[type=checkbox] { cursor: pointer; }
  .color-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .norm-btn { padding: 2px 10px; font-size: 0.82em; font-weight: 600; cursor: pointer; border-radius: 4px; border: 1px solid #ccc; min-width: 78px; }
  .norm-btn.on  { background: #d4edda; color: #155724; border-color: #c3e6cb; }
  .norm-btn.off { background: #f8f9fa; color: #6c757d; border-color: #dee2e6; }
  .formula { font-size: 0.82em; color: #888; font-family: 'Courier New', monospace; }

  .bottom-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
  .plots { display: flex; gap: 16px; }
  .plots > div { flex: 1; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 8px; }
  @media (max-width: 900px) { .plots { flex-direction: column; } }
</style>
</head>
<body>
<h1>GD Trajectory Dashboard</h1>

<div class="controls">
  <div class="top-row">
    <div><label>n:</label><select id="nSelect" class="ctrl-select"></select></div>
    <div><label>d:</label><select id="dSelect" class="ctrl-select"></select></div>
    <div><label>Seed:</label><select id="seedSelect" class="ctrl-select"></select></div>
  </div>

  <div class="slider-row">
    <label>k/n ratio:</label>
    <button class="step-btn" id="btnPrev">&lt;</button>
    <input type="range" id="ratioSlider" min="0" max="0" value="0" step="1">
    <button class="step-btn" id="btnNext">&gt;</button>
    <span class="slider-label" id="sliderLabel">—</span>
  </div>

  <table class="metric-table" id="metricTable"></table>

  <div class="bottom-row">
    <div><label>X-axis:</label> <button class="step-btn" id="btnScale" style="width:auto;padding:0 10px;font-size:0.9em;font-weight:600;">Log</button></div>
    <div style="display:flex;align-items:center;gap:6px;">
      <label>t range:</label>
      <input type="text" id="tMin" class="ctrl-select" style="width:80px;" placeholder="min" value="0"> —
      <input type="text" id="tMax" class="ctrl-select" style="width:80px;" placeholder="max">
      <button class="step-btn" id="btnApplyRange" style="width:auto;padding:0 8px;font-size:0.8em;">Apply</button>
      <button class="step-btn" id="btnResetRange" style="width:auto;padding:0 8px;font-size:0.8em;">Reset</button>
    </div>
  </div>
</div>

<div class="plots">
  <div><div id="plotLeft"  style="width:100%;height:500px;"></div></div>
  <div><div id="plotRight" style="width:100%;height:500px;"></div></div>
</div>

<script>
const DATA = __JSON_DATA__;

const METRICS = [
  { key: 'norm',          label: 'Norm',         color: '#2ca02c', timesKey: 'times',         defaultNorm: true,  defaultVisible: false, defaultArgmin: true,  formula: '(x \u2212 min) / (max \u2212 min)' },
  { key: 'norm_diff',     label: '| ||w_t|| \u2212 ||w*_{0:k}|| |', color: '#17becf', timesKey: 'times', defaultNorm: true, defaultVisible: true,  defaultArgmin: true, formula: 'x / max(x)  (no shift, x\u22650)', computed: true, normFn: 'divAbsMax' },
  { key: 'train_loss',    label: 'Train Loss',   color: '#d62728', timesKey: 'loss_times',    defaultNorm: true,  defaultVisible: false, defaultArgmin: true,  formula: '(x \u2212 min) / (max \u2212 min)' },
  { key: 'pop_loss',      label: 'Test Loss',    color: '#9467bd', timesKey: 'pop_loss_times', defaultNorm: true, defaultVisible: true,  defaultArgmin: true,  formula: '(x \u2212 min) / (max \u2212 min)' },
  { key: 'angle_w_star',  label: 'Angle to w\u204e',color: '#1f77b4', timesKey: 'times',     defaultNorm: true,  defaultVisible: true,  defaultArgmin: true,  formula: 'x / 180  (no shift, 0\u00b0\u21920, 180\u00b0\u21921)', normFn: 'div180' },
  { key: 'angle_w_tilde', label: 'Angle to w\u0303', color: '#ff7f0e', timesKey: 'times',    defaultNorm: true,  defaultVisible: false, defaultArgmin: true,  formula: 'x / 180  (no shift, 0\u00b0\u21920, 180\u00b0\u21921)', normFn: 'div180' },
];

// Per-metric state
const state = {};
METRICS.forEach(m => { state[m.key] = { visible: m.defaultVisible, normalize: m.defaultNorm, showArgmin: m.defaultArgmin }; });

// Build metric table
const table = document.getElementById('metricTable');
METRICS.forEach(m => {
  const tr = document.createElement('tr');
  // Col 1: checkbox + color dot + label
  const td1 = document.createElement('td');
  const nameDiv = document.createElement('div');
  nameDiv.className = 'metric-name';
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = m.defaultVisible;
  cb.addEventListener('change', () => { state[m.key].visible = cb.checked; updatePlots(); });
  const dot = document.createElement('span');
  dot.className = 'color-dot'; dot.style.background = m.color;
  nameDiv.appendChild(cb); nameDiv.appendChild(dot);
  nameDiv.appendChild(document.createTextNode(m.label));
  td1.appendChild(nameDiv);
  // Col 2: normalize toggle
  const td2 = document.createElement('td');
  const btn = document.createElement('button');
  btn.className = 'norm-btn ' + (m.defaultNorm ? 'on' : 'off');
  btn.textContent = 'Normalize: ' + (m.defaultNorm ? 'ON' : 'OFF');
  btn.addEventListener('click', () => {
    state[m.key].normalize = !state[m.key].normalize;
    btn.className = 'norm-btn ' + (state[m.key].normalize ? 'on' : 'off');
    btn.textContent = 'Normalize: ' + (state[m.key].normalize ? 'ON' : 'OFF');
    updatePlots();
  });
  td2.appendChild(btn);
  // Col 3: formula
  const td3 = document.createElement('td');
  td3.className = 'formula'; td3.textContent = m.formula;
  // Col 4: argmin toggle
  const td4 = document.createElement('td');
  const abtn = document.createElement('button');
  abtn.className = 'norm-btn ' + (m.defaultArgmin ? 'on' : 'off');
  abtn.textContent = 'Argmin: ' + (m.defaultArgmin ? 'ON' : 'OFF');
  abtn.addEventListener('click', () => {
    state[m.key].showArgmin = !state[m.key].showArgmin;
    abtn.className = 'norm-btn ' + (state[m.key].showArgmin ? 'on' : 'off');
    abtn.textContent = 'Argmin: ' + (state[m.key].showArgmin ? 'ON' : 'OFF');
    updatePlots();
  });
  td4.appendChild(abtn);
  tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4);
  table.appendChild(tr);
});

// Selects
const slider    = document.getElementById('ratioSlider');
const sliderLabel = document.getElementById('sliderLabel');
const nSelect   = document.getElementById('nSelect');
const dSelect   = document.getElementById('dSelect');
const seedSelect= document.getElementById('seedSelect');
const btnScale  = document.getElementById('btnScale');
const tMin      = document.getElementById('tMin');
const tMax      = document.getElementById('tMax');
let xLog = true;
let prevRatioIdx = null, currentRatioIdx = 0;
let filteredRatios = [], filteredRuns = {};

const allN = [...new Set(DATA.runs.map(r => r.n))].sort((a,b)=>a-b);
const allD = [...new Set(DATA.runs.map(r => r.d))].sort((a,b)=>a-b);
allN.forEach(n => { const o=document.createElement('option'); o.value=n; o.textContent=n; nSelect.appendChild(o); });
allD.forEach(d => { const o=document.createElement('option'); o.value=d; o.textContent=d; dSelect.appendChild(o); });
const ndCounts = {};
DATA.runs.forEach(r => { const k=r.n+','+r.d; ndCounts[k]=(ndCounts[k]||0)+1; });
const bestND = Object.entries(ndCounts).sort((a,b)=>b[1]-a[1])[0][0].split(',');
nSelect.value = bestND[0]; dSelect.value = bestND[1];

nSelect.addEventListener('change', onNDChange);
dSelect.addEventListener('change', onNDChange);
seedSelect.addEventListener('change', updatePlots);
slider.addEventListener('input', onSliderChange);
document.getElementById('btnPrev').addEventListener('click', () => { slider.value = Math.max(0, parseInt(slider.value)-1); onSliderChange(); });
document.getElementById('btnNext').addEventListener('click', () => { slider.value = Math.min(parseInt(slider.max), parseInt(slider.value)+1); onSliderChange(); });
document.getElementById('btnApplyRange').addEventListener('click', updatePlots);
document.getElementById('btnResetRange').addEventListener('click', () => { tMin.value='0'; tMax.value=''; updatePlots(); });
tMin.addEventListener('keydown', e => { if(e.key==='Enter') updatePlots(); });
tMax.addEventListener('keydown', e => { if(e.key==='Enter') updatePlots(); });
btnScale.addEventListener('click', () => { xLog=!xLog; btnScale.textContent=xLog?'Log':'Linear'; updatePlots(); });

function round(v) { return Math.round(v*1e6)/1e6; }

function normalize(arr) {
  const mn=Math.min(...arr), mx=Math.max(...arr);
  if(mx===mn) return arr.map(()=>0.5);
  return arr.map(v=>(v-mn)/(mx-mn));
}

function parseValue(str) {
  if(!str||str.trim()==='') return null;
  str=str.trim().toLowerCase();
  const mult={'k':1e3,'m':1e6}, last=str[str.length-1];
  if(mult[last]) return parseFloat(str.slice(0,-1))*mult[last];
  return parseFloat(str);
}

function getXRange() {
  const lo=parseValue(tMin.value), hi=parseValue(tMax.value);
  if(lo===null&&hi===null) return undefined;
  if(xLog){
    const logLo=lo!==null?Math.log10(Math.max(lo,1)):null;
    const logHi=hi!==null?Math.log10(hi):null;
    if(logLo!==null&&logHi!==null) return [logLo,logHi];
    if(logLo!==null) return [logLo,undefined];
    return [undefined,logHi];
  }
  return [lo,hi];
}

function plotLayout(title) {
  return {
    title: { text: title, font: { size: 13 } },
    xaxis: { title: 'Iteration t', type: xLog?'log':'linear', gridcolor: '#eee', range: getXRange() },
    yaxis: { gridcolor: '#eee' },
    legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' },
    margin: { t: 50, b: 80, l: 50, r: 20 },
    plot_bgcolor: '#fafafa',
    hovermode: 'closest',
    shapes: [],
  };
}

function buildTracesAndLayout(run, title) {
  if(!run) return { traces: [], layout: plotLayout(title) };
  const traces = [];
  const layout = plotLayout(title);

  METRICS.forEach(m => {
    if(!state[m.key].visible) return;
    // Compute derived metrics
    if(m.computed) {
      if(m.key==='norm_diff' && run.norm && run.w_star_norm!=null) {
        run._norm_diff = run.norm.map(v => Math.abs(v - run.w_star_norm));
      }
    }
    const raw = m.computed ? run['_'+m.key] : run[m.key];
    if(!raw) return;
    const times = run[m.timesKey] || run.times;
    const y = state[m.key].normalize
      ? (m.normFn==='div180'     ? raw.map(v=>v/180)
       : m.normFn==='divAbsMax'  ? (v => { const mx=Math.max(...raw.map(Math.abs)); return mx>0 ? raw.map(x=>x/mx) : raw; })()
       : normalize(raw))
      : raw;
    traces.push({
      x: times, y,
      type: 'scatter', mode: 'lines',
      name: m.label,
      line: { color: m.color, width: 2 },
      text: raw.map(v => m.label + ' = ' + v.toFixed(4)),
      hovertemplate: '%{text}<br>t = %{x}<extra></extra>',
    });
    // Argmin vertical line
    if(state[m.key].showArgmin) {
      const minIdx = raw.reduce((best, v, i) => v < raw[best] ? i : best, 0);
      const argminT = times[minIdx];
      layout.shapes.push({
        type:'line', xref:'x', yref:'paper',
        x0: argminT, x1: argminT, y0:0, y1:1,
        line:{color: m.color, width:1.5, dash:'dot'},
      });
      traces.push({
        x:[null],y:[null],type:'scatter',mode:'lines',
        name:'argmin('+m.label+') = t'+argminT,
        line:{color:m.color,width:1.5,dash:'dot'},
        showlegend:true,
      });
    }
    // w* norm reference line (only for norm metric, only when normalized)
    if(m.key==='norm' && state[m.key].normalize && run.w_star_norm!=null) {
      const mn=Math.min(...raw), mx=Math.max(...raw);
      if(mx!==mn) {
        const wStarNormed = (run.w_star_norm-mn)/(mx-mn);
        traces.push({
          x:[times[0],times[times.length-1]], y:[wStarNormed,wStarNormed],
          type:'scatter', mode:'lines',
          name:'||w*_{0:'+run.k+'}||',
          line:{color:'gray',width:1.5,dash:'dash'},
          hovertemplate:'||w*_{0:'+run.k+'}|| = '+run.w_star_norm.toFixed(4)+'<extra></extra>',
        });
      }
    }
  });

  // Stopping time as layout shape (spans full plot height regardless of y scale)
  if(run.stopping_times) {
    run.stopping_times.forEach((st, i) => {
      layout.shapes.push({
        type:'line', xref:'x', yref:'paper',
        x0: st, x1: st,
        y0:0, y1:1,
        line:{color:'red',width:1.5,dash:'dash'},
      });
      if(i===0) traces.push({
        x:[null],y:[null],type:'scatter',mode:'lines',
        name:'t* = '+st, line:{color:'red',width:1.5,dash:'dash'},
        showlegend:true,
      });
    });
  }
  return { traces, layout };
}

function makeTitle(run, prefix) {
  if(!run) return prefix+': (no data)';
  const ratio=round(run.k/run.n);
  return prefix+': k/n='+ratio.toFixed(4)+'  (k='+run.k+', n='+run.n+', d='+run.d+', seed='+run.seed+')';
}

function onNDChange() {
  const n=parseInt(nSelect.value), d=parseInt(dSelect.value);
  filteredRuns={};
  DATA.runs.forEach(r => {
    if(r.n!==n||r.d!==d) return;
    const ratio=round(r.k/r.n);
    if(!filteredRuns[ratio]) filteredRuns[ratio]=[];
    filteredRuns[ratio].push(r);
  });
  filteredRatios=Object.keys(filteredRuns).map(Number).sort((a,b)=>a-b);
  prevRatioIdx=null; currentRatioIdx=0;
  slider.min=0; slider.max=Math.max(0,filteredRatios.length-1); slider.value=0;
  populateSeeds(0); updateSliderLabel(); updatePlots();
}

function populateSeeds(ratioIdx) {
  const runs=getRunsForRatio(ratioIdx), prevSeed=seedSelect.value;
  seedSelect.innerHTML='';
  runs.forEach((run,i) => {
    const o=document.createElement('option'); o.value=i; o.textContent='seed='+run.seed; seedSelect.appendChild(o);
  });
  const matchIdx=runs.findIndex(r=>String(r.seed)===prevSeed);
  seedSelect.value=matchIdx>=0?matchIdx:0;
}

function getRunsForRatio(ratioIdx) {
  if(ratioIdx<0||ratioIdx>=filteredRatios.length) return [];
  return filteredRuns[filteredRatios[ratioIdx]]||[];
}

function getRun(ratioIdx) {
  const runs=getRunsForRatio(ratioIdx);
  if(runs.length===0) return null;
  if(ratioIdx!==currentRatioIdx){
    const cur=getRunsForRatio(currentRatioIdx);
    const selIdx=parseInt(seedSelect.value)||0;
    if(cur[selIdx]){const match=runs.find(r=>r.seed===cur[selIdx].seed); if(match) return match;}
  }
  return runs[Math.min(parseInt(seedSelect.value)||0,runs.length-1)];
}

function updateSliderLabel() {
  const run=getRun(currentRatioIdx);
  sliderLabel.textContent = run ? 'k/n = '+round(run.k/run.n).toFixed(4)+'  (k='+run.k+')' : '—';
}

function updatePlots() {
  const cur=getRun(currentRatioIdx);
  const {traces:tL, layout:lL}=buildTracesAndLayout(cur, makeTitle(cur,'Current'));
  Plotly.react('plotLeft', tL, lL, {responsive:true});

  if(prevRatioIdx!==null&&prevRatioIdx!==currentRatioIdx){
    const prev=getRun(prevRatioIdx);
    const {traces:tR, layout:lR}=buildTracesAndLayout(prev, makeTitle(prev,'Previous'));
    Plotly.react('plotRight', tR, lR, {responsive:true});
  } else {
    Plotly.react('plotRight', [], plotLayout('Previous: (none)'), {responsive:true});
  }
}

function onSliderChange() {
  const newIdx=parseInt(slider.value);
  if(newIdx!==currentRatioIdx){ prevRatioIdx=currentRatioIdx; currentRatioIdx=newIdx; }
  populateSeeds(currentRatioIdx); updateSliderLabel(); updatePlots();
}

onNDChange();
</script>
</body>
</html>
"""

# Inject data
html = HTML.replace('__JSON_DATA__', json_blob)
out = os.path.join(os.path.dirname(__file__), "../dashboard_2/dashboard.html")
with open(out, "w") as f:
    f.write(html)
print(f"Written {out} ({len(html)/1024:.1f} KB)")
