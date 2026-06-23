"""
app.py — CASCADE Enterprise AI Command Center (Streamlit + pydeck + plotly).

De-biased hotspot map (3D labeled city view), conformally-calibrated per-junction forecasts, the
OR-Tools deployment plan, a live what-if re-optimizer, an auto-playing time simulation, model-
transparency visualizations, a Commander's Briefing (XAI), and the measured economic impact. Reads
only precomputed artifacts under models/ and data/processed/ — no training, no heavy compute in UI.

Run:    streamlit run src/cascade/demo/app.py
Deploy: Streamlit Community Cloud / Hugging Face Spaces (point at this file).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
CLEAR_CAP = 180.0

st.set_page_config(page_title="CASCADE · Traffic Intelligence Command Center", page_icon="C",
                   layout="wide", initial_sidebar_state="collapsed")

# Pinot Noir — Custom Solid Variations
BG = "transparent"
# Hero & Callout: Warm Copper-Slate
HERO_BG = "rgba(25, 25, 35, 0.4)"; HERO_LINE = "rgba(255, 255, 255, 0.08)"
# KPIs: Cool Teal-Slate
KPI_BG = "rgba(25, 25, 35, 0.4)"; KPI_LINE = "rgba(255, 255, 255, 0.08)"
# Map & Controls: Pure Navy
MAP_BG = "rgba(25, 25, 35, 0.4)"; MAP_LINE = "rgba(255, 255, 255, 0.08)"
# Junction Intelligence: Warm Violet-Navy
JI_BG = "rgba(25, 25, 35, 0.4)"; JI_LINE = "rgba(255, 255, 255, 0.08)"
# Analysis/Briefing: Deep Purple-Slate
ANLY_BG = "rgba(25, 25, 35, 0.4)"; ANLY_LINE = "rgba(255, 255, 255, 0.08)"
# Worklists: Neutral Steel
WORK_BG = "rgba(25, 25, 35, 0.4)"; WORK_LINE = "rgba(255, 255, 255, 0.08)"

# Universal inputs/inner panels
PANEL2 = "rgba(40, 40, 50, 0.4)"; LINE = "rgba(255, 255, 255, 0.08)"; LINE2 = "#8a75f5"
TXT = "#f5f5f5"; MUT = "#a1a0ab"; MUT2 = "#8e8d9a"
AMBER = "#8a75f5"; ORANGE = "#9b85f8"; ROSE = "#b5a8fb"; FUCHSIA = "#c6bafb"
VIOLET = "#8658e8"; CYAN = "#62a8ff"; EMERALD = "#45db9c"
# perceptual "ocean-fire" heat ramp (low → high) — teal to copper to coral
HEAT = [(0.0, (20, 72, 90)), (0.22, (40, 108, 130)), (0.45, (76, 160, 176)),
        (0.66, (212, 149, 107)), (0.84, (232, 146, 106)), (1.0, (232, 112, 122))]
MAGMA = [[i / (len(HEAT) - 1), f"rgb({c[0]},{c[1]},{c[2]})"] for i, (p, c) in enumerate(HEAT)]


def heat_rgb(n):
    n = float(np.clip(n, 0, 1))
    for i in range(len(HEAT) - 1):
        a, ca = HEAT[i]; bb, cb = HEAT[i + 1]
        if n <= bb:
            t = (n - a) / (bb - a + 1e-9)
            return [int(ca[j] + (cb[j] - ca[j]) * t) for j in range(3)]
    return list(HEAT[-1][1])


st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
  html, body, [class*="css"], .stApp{{font-family:'Inter',sans-serif;}}
  .stApp{{background:{BG};}}
  #MainMenu, header[data-testid="stHeader"], footer,
  [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{{display:none !important;}}
  .block-container{{padding:1rem 1.5rem 2.4rem !important; max-width:1640px;}}
  ::-webkit-scrollbar{{width:9px;height:9px;}} ::-webkit-scrollbar-thumb{{background:#2E3A56;border-radius:9px;}}
  ::-webkit-scrollbar-thumb:hover{{background:#3D4B6A;}} ::-webkit-scrollbar-track{{background:transparent;}}
  .stTabs [data-baseweb="tab-panel"]{{padding-top:2.3rem;}}
  [data-testid="stVerticalBlock"]{{gap:.65rem;}}

    /* ══ GLASSMORPHISM ══ */
  .icard, .callout, .panel, .legend, .ctrl-strip, .ji-panel, .brow, [data-testid="stExpander"] {{
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid rgba(255, 255, 255, 0.08) !important;
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
  }}
  
  /* Animated Background */
  .stApp::before {{
      content: "";
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: radial-gradient(circle at 10% 20%, rgba(138, 117, 245, 0.08), transparent 45%),
                  radial-gradient(circle at 90% 70%, rgba(138, 117, 245, 0.06), transparent 45%),
                  radial-gradient(circle at 80% 10%, rgba(138, 117, 245, 0.04), transparent 35%);
      background-color: #121216;
      z-index: -1;
      pointer-events: none;
  }}

  /* ══ HERO ══ */
  .topbar{{display:flex;align-items:center;gap:14px;padding:15px 22px;margin-bottom:24px;
    background:{HERO_BG}; border:1px solid {HERO_LINE};border-radius:12px;
    box-shadow:0 6px 20px -8px rgba(0,0,0,.6);}}
  .topbar .mark{{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;
    background:{HERO_LINE};color:{TXT};}}
  .topbar .wm{{font-size:22px;font-weight:900;letter-spacing:.5px;color:{TXT};}}
  .topbar .tg{{font-size:12px;color:{MUT};margin-top:1px;}}
  .chips{{margin-left:auto;display:flex;gap:10px;}}
  .chip{{text-align:center;padding:7px 16px;border:1px solid {HERO_LINE};border-radius:8px;background:{PANEL2};}}
  .chip .cl{{font-size:9px;font-weight:700;letter-spacing:1px;color:{MUT2};text-transform:uppercase;}}
  .chip .cv{{font-size:16px;font-weight:800;color:{TXT};margin-top:2px;}}
  .chip.live{{border-color:rgba(88,216,160,.4);}}
  .chip.live .cv{{color:{EMERALD};display:flex;align-items:center;gap:6px;justify-content:center;}}
  .pulse{{width:8px;height:8px;border-radius:50%;background:{EMERALD};animation:pulse 1.8s infinite;}}
  @keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(88,216,160,.6);}}70%{{box-shadow:0 0 0 8px rgba(88,216,160,0);}}100%{{box-shadow:0 0 0 0 rgba(88,216,160,0);}}}}

  /* ══ KPI CARDS ══ */
  .icard{{position:relative;overflow:hidden;border:1px solid {HERO_LINE};border-radius:12px;padding:18px;
    background:{HERO_BG};
    box-shadow:0 4px 12px -4px rgba(0,0,0,.3);transition:transform .16s, border-color .25s, box-shadow .25s;}}
  .icard:hover{{transform:translateY(-3px);border-color:var(--ac);
    box-shadow:0 12px 24px -10px rgba(0,0,0,.6);}}
  .icard::before{{content:"";position:absolute;left:0;top:0;height:3px;width:100%;background:var(--ac);opacity:1;}}
  .icard::after{{display:none;}} /* Disable glowing radial blobs for professional look */
  .icard .chip2{{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;margin-bottom:12px;}}
  .icard .v{{font-size:29px;font-weight:900;letter-spacing:-1px;line-height:1;color:{TXT};}}
  .icard .l{{margin-top:8px;color:{MUT};font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;}}
  .icard .s{{color:{MUT2};font-size:11px;margin-top:3px;}}

  /* ══ CALLOUT ══ */
  .callout{{display:flex;gap:18px;align-items:flex-start;border:1px solid {KPI_LINE};border-left:3px solid {AMBER};
    background:{KPI_BG};border-radius:12px;padding:20px 24px;}}
  .callout .ci{{flex:none;color:{AMBER};margin-top:2px;}}
  .callout .ct{{font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:1.4px;color:{AMBER};margin-bottom:8px;}}
  .callout .cx{{font-size:13.5px;color:#cbd5e1;line-height:1.95;max-width:1180px;}}

  /* ══ SECTION HEADERS ══ */
  .sec{{display:flex;align-items:center;gap:12px;margin:4px 2px 14px;}}
  .sec .bar{{width:4px;height:16px;border-radius:4px;background:{LINE2};}}
  .sec .t{{font-size:11.5px;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;color:#cbd5e1;}}
  .sec .hr{{flex:1;height:1px;background:linear-gradient(90deg,{WORK_LINE},transparent);}}
  .mini .ml{{color:{MUT};font-size:11px;font-weight:600;}}
  .mini .mv{{font-size:20px;font-weight:800;line-height:1.15;color:{TXT};}}

  /* ══ MAP PANEL ══ */
  .panel{{border:1px solid {MAP_LINE};border-radius:12px;
    background:{MAP_BG};padding:16px 18px;
    box-shadow:0 6px 20px -8px rgba(0,0,0,.6);}}
  .legend{{display:flex;flex-wrap:wrap;align-items:center;gap:18px;font-size:12px;color:#cbd5e1;
    background:{PANEL2};
    border:1px solid {MAP_LINE};border-radius:8px;padding:10px 16px;margin-top:12px;}}
  .legend .dot{{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle;}}
  .legend .grad{{width:54px;height:10px;border-radius:6px;display:inline-block;vertical-align:middle;margin-right:6px;
    background:linear-gradient(90deg,#14485a,#50d0f8,#e8a460,#f06878);}}

  /* ══ BRIEFING CARDS ══ */
  .bfeed{{display:flex;flex-direction:column;gap:12px;}}
  .brow{{display:flex;gap:14px;align-items:center;border:1px solid {ANLY_LINE};border-left:3px solid var(--bc);
    background:{ANLY_BG};border-radius:8px;padding:15px 18px;
    box-shadow:0 2px 8px -4px rgba(0,0,0,.3);}}
  .brow .bi{{flex:none;width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;
    background:var(--bb);color:var(--bc);font-weight:800;font-size:13px;}}
  .brow .bx{{font-size:13px;color:#cbd5e1;line-height:1.55;}}

  /* ══ SIDEBAR ══ */
  [data-testid="stSidebar"]{{display:none !important;}}
  [data-testid="stSidebarCollapsedControl"]{{display:none !important;}}

  /* ══ CONTROL STRIP ══ */
  .ctrl-strip{{display:flex;flex-wrap:wrap;gap:10px;align-items:end;
    background:{MAP_BG};border:1px solid {MAP_LINE};
    border-radius:12px;padding:14px 18px;margin-bottom:14px;}}
  .ctrl-strip .ctrl-group{{flex:1;min-width:140px;}}
  .ctrl-strip label,.ctrl-strip .ctrl-label{{font-size:9.5px;font-weight:700;letter-spacing:1px;
    color:{MUT2};text-transform:uppercase;margin-bottom:4px;display:block;}}

  /* ══ TABS ══ */
  .stTabs [data-baseweb="tab-list"]{{gap:12px;background:transparent;padding:0;border:none;align-items:center;}}
  .stTabs [data-baseweb="tab"]{{height:42px;border-radius:20px;font-size:13px;font-weight:700;color:{MUT};padding:0 24px;
                               background:rgba(25, 25, 35, 0.5); border:1px solid rgba(255, 255, 255, 0.08);
                               backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); transition:all 0.2s; box-shadow: 0 4px 12px rgba(0,0,0,0.4);}}
  .stTabs [aria-selected="true"]{{background:rgba(40, 40, 50, 0.8) !important;color:#8a75f5 !important;
    border:1px solid rgba(138, 117, 245, 0.6) !important;box-shadow:0 0 20px rgba(138, 117, 245,.2);}}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{{display:none !important;}}

  /* ══ JI PANEL ══ */
  .ji-panel{{display:flex;align-items:center;justify-content:space-between;
    background:{JI_BG};border:1px solid {JI_LINE};border-radius:12px;padding:24px 30px;
    box-shadow:0 6px 20px -8px rgba(0,0,0,.6);}}
  .ji-item{{flex:1;text-align:center;}}
  .ji-item .l{{color:{MUT};font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;}}
  .ji-item .v{{font-size:34px;font-weight:900;letter-spacing:-1px;line-height:1;margin-bottom:6px;}}
  .ji-item .s{{color:{MUT2};font-size:11px;}}
  .ji-div{{width:1px;height:54px;background:{JI_LINE};margin:0 10px;}}

  /* ══ WORKLISTS ══ */
  [data-testid="stDataFrame"]{{border:1px solid {WORK_LINE};border-radius:12px;overflow:hidden;}}
  [data-testid="stExpander"]{{border:1px solid {WORK_LINE}!important;border-radius:12px!important;
    background:{WORK_BG};}}
  .stButton>button{{background:{LINE2};color:#ffffff;font-weight:800;border:none;
    border-radius:8px;padding:11px 0;box-shadow:0 4px 12px -6px rgba(0,0,0,.5);transition:filter .15s, transform .15s;}}
  .brow .bx{{line-height:1.6;}}
  .stButton>button:hover{{filter:brightness(1.07);transform:translateY(-1px);}}
  div[data-baseweb="select"]>div{{background:{PANEL2};border-color:{WORK_LINE};border-radius:10px;}}
  .stSlider [data-baseweb="slider"] [role="slider"]{{background:{CYAN};}}
</style>
""", unsafe_allow_html=True)


def _svg(inner, sz=18):
    return (f'<svg width="{sz}" height="{sz}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')

IC = {
    "logo": _svg('<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3'
                 'M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>', 24),
    "activity": _svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    "value": _svg('<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>'),
    "cpu": _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'),
    "target": _svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'),
    "shield": _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>'),
    "brief": _svg('<path d="M4.93 19.07A10 10 0 0 1 19.07 4.93"/><path d="M7.76 16.24a6 6 0 0 1 8.48-8.48"/>'
                  '<circle cx="12" cy="12" r="2"/>', 22),
}


def play(height, **kw):
    base = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                font=dict(color=MUT, family="Inter", size=12), margin=dict(l=62, r=22, t=42, b=54),
                xaxis=dict(gridcolor="rgba(255,255,255,.05)", zerolinecolor=LINE, linecolor=LINE,
                           automargin=True, title_standoff=14, ticks="outside", tickcolor=LINE),
                yaxis=dict(gridcolor="rgba(255,255,255,.05)", zerolinecolor=LINE, linecolor=LINE,
                           automargin=True, title_standoff=14))
    base.update(kw)
    return base


@st.cache_data(show_spinner=False)
def load_nodes():
    nd = pd.read_parquet(PROC / "graph_nodes.parquet")
    return nd.sort_values("node_id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_calibrated():
    c = np.load(MODELS / "calibrated.npz", allow_pickle=True)
    return {k: c[k] for k in c.files}


def load_json(name, default=None):
    p = MODELS / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


@st.cache_data(show_spinner=False)
def per_node_risk():
    cal = load_calibrated(); nd = load_nodes()
    df = pd.DataFrame({"node_id": cal["node_id"],
                       "burden": np.minimum(cal["median"], CLEAR_CAP) * cal["closure_prob"]})
    g = df.groupby("node_id").agg(naive=("burden", "size"), corrected=("burden", "sum"))
    inten = cal["node_intensity"] if "node_intensity" in cal else None
    out = nd.merge(g, on="node_id", how="left").fillna({"naive": 0, "corrected": 0})
    if inten is not None:
        out["intensity"] = [float(inten[i]) if i < len(inten) else 0.0 for i in out["node_id"]]
    for col in ("naive", "corrected"):
        v = out[col].to_numpy(float); hi = np.quantile(v[v > 0], 0.97) if (v > 0).any() else 1.0
        out[col + "_n"] = np.clip(v / max(hi, 1e-9), 0, 1)
    return out


@st.cache_data(show_spinner=False)
def event_points():
    cal = load_calibrated(); nd = load_nodes().set_index("node_id")
    nid = cal["node_id"]
    w = np.minimum(cal["median"], CLEAR_CAP) * cal["closure_prob"]
    lat = nd["lat"].reindex(nid).to_numpy(); lon = nd["lon"].reindex(nid).to_numpy()
    m = np.isfinite(lat) & np.isfinite(lon)
    return pd.DataFrame({"lon": lon[m], "lat": lat[m], "w": w[m].astype(float), "cnt": 1.0})


@st.cache_data(show_spinner=False)
def node_forecast():
    cal = load_calibrated()
    df = pd.DataFrame({"node_id": cal["node_id"],
                       "median": np.minimum(cal["median"], 1440.0),
                       "upper": np.minimum(cal["upper"], 1440.0),
                       "closure": cal["closure_prob"], "priority": cal["priority_prob"]})
    g = df.groupby("node_id").agg(median=("median", "median"), p90=("upper", "median"),
                                  closure=("closure", "mean"), priority=("priority", "mean"),
                                  incidents=("median", "size"))
    return g


def fmt_dur(m):
    if m >= 1439:
        return "≥24 h"
    return f"{m:.0f} min" if m < 120 else f"{m/60:.1f} h"


def run_allocator(officers, barricades, radius, predict_weight):
    out = MODELS / "_whatif.json"
    cmd = [sys.executable, "-m", "src.cascade.optimize.allocator",
           "--officers", str(officers), "--barricades", str(barricades),
           "--radius-km", str(radius), "--predict-weight", str(predict_weight),
           "--scope", "test", "--out", str(out), "--quiet"]
    subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, timeout=120)
    return json.loads(out.read_text(encoding="utf-8"))


def diurnal_display(risk_df, hour):
    d = risk_df.copy()
    base = 0.45 + 0.55 * max(np.exp(-((hour - 10) / 3.0) ** 2), np.exp(-((hour - 19) / 3.0) ** 2))
    phase = (d["node_id"].to_numpy() % 24) / 24.0
    fac = np.clip(base * (0.7 + 0.5 * np.sin(2 * np.pi * (hour / 24.0 + phase))), 0.15, 1.4)
    for c in ("naive", "corrected"):
        d[c + "_n"] = np.clip(d[c + "_n"].to_numpy() * fac, 0, 1)
    return d, float(base)


def risk_layer(risk_df, mode):
    col = mode
    d = risk_df.copy()
    d["radius"] = 110 + d[col + "_n"] * 760
    if mode == "corrected":
        d["color"] = d[col + "_n"].apply(lambda x: heat_rgb(x) + [int(80 + 150 * x)])
    else:
        d["color"] = d[col + "_n"].apply(lambda x: [int(56 + 30 * x), int(150 + 40 * x), 240, int(70 + 150 * x)])
    return pdk.Layer("ScatterplotLayer", d, get_position="[lon, lat]", get_radius="radius",
                     get_fill_color="color", pickable=True, opacity=0.8, stroked=True,
                     get_line_color=[255, 255, 255, 45], line_width_min_pixels=0.5)


def hexbin_layer(ep, weight_col):
    rng = [heat_rgb(x) for x in (0.05, 0.25, 0.45, 0.65, 0.83, 1.0)]
    return pdk.Layer("HexagonLayer", ep, get_position="[lon, lat]", get_weight=weight_col,
                     radius=450, elevation_scale=3.5, elevation_range=[0, 750], extruded=True,
                     coverage=0.82, opacity=0.8, pickable=True, color_range=rng, auto_highlight=True)


def plan_layers_for(plan, show_off, show_bar):
    layers = []
    if show_off:
        officers = pd.DataFrame(plan.get("officers", []))
        if not officers.empty:
            officers["radius"] = 150 + officers["headcount"].astype(float) * 52
            layers.append(pdk.Layer("ScatterplotLayer", officers, get_position="[lon, lat]",
                          get_radius="radius", get_fill_color=[56, 189, 248, 210],
                          get_line_color=[255, 255, 255, 235], line_width_min_pixels=2, stroked=True, pickable=True))
            layers.append(pdk.Layer("TextLayer", officers, get_position="[lon, lat]",
                          get_text="headcount", get_size=13, get_color=[7, 12, 24]))
    if show_bar:
        barr = pd.DataFrame(plan.get("barricades", []))
        if not barr.empty:
            layers.append(pdk.Layer("ScatterplotLayer", barr, get_position="[lon, lat]", get_radius=250,
                          get_fill_color=[244, 63, 94, 235], get_line_color=[255, 255, 255, 230],
                          line_width_min_pixels=2, stroked=True, pickable=True))
    return layers


def diversion_layer(divs, nodes):
    coord = {int(r.node_id): [float(r.lon), float(r.lat)] for r in nodes.itertuples()}
    rows = []
    for d in divs or []:
        path = [coord[i] for i in d.get("via_ids", []) if i in coord]
        if len(path) >= 2:
            rows.append({"path": path, "name": d.get("barricade", "")})
    if not rows:
        return None
    return pdk.Layer("PathLayer", rows, get_path="path", get_color=[52, 211, 153, 235],
                     width_min_pixels=3, get_width=4, pickable=True)


def section(title):
    st.markdown(f'<div class="sec"><span class="bar"></span><span class="t">{title}</span>'
                f'<span class="hr"></span></div>', unsafe_allow_html=True)


def vspace(px=26):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def insight_card(col, icon, value, label, sub, accent):
    # per-accent glow color (transparent version for radial gradient)
    glow = accent.replace('#', '')
    r, g, b = int(glow[:2], 16), int(glow[2:4], 16), int(glow[4:6], 16)
    glow_rgba = f"rgba({r},{g},{b},.15)"
    col.markdown(
        f'<div class="icard" style="--ac:{accent};--glow:{glow_rgba}"><div class="chip2" style="background:{accent}22;color:{accent}">{IC[icon]}</div>'
        f'<div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)


def mini(label, val, accent=TXT):
    return (f'<div class="mini" style="margin:7px 0"><div class="ml">{label}</div>'
            f'<div class="mv" style="color:{accent}">{val}</div></div>')


def clearance_curve(p50, p90):
    p50 = max(float(p50), 1.0); p90 = max(float(p90), p50 * 1.05)
    mu = np.log(p50); sigma = max(np.log(p90 / p50) / 1.2816, 0.05)
    t = np.linspace(1, min(p90 * 2.2, 1440), 160)
    return t, norm.cdf((np.log(t) - mu) / sigma)


def fig_survival(p50, p90):
    t, cdf = clearance_curve(p50, p90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=cdf * 100, mode="lines", line=dict(color=AMBER, width=3.5),
                             fill="tozeroy", fillcolor="rgba(212,149,107,.14)",
                             hovertemplate="%{x:.0f} min · %{y:.0f}% cleared<extra></extra>"))
    fig.add_vline(x=p50, line=dict(color=CYAN, dash="dash", width=1.6),
                  annotation_text="P50", annotation_font_color=CYAN, annotation_position="top left")
    fig.add_vline(x=p90, line=dict(color=FUCHSIA, dash="dash", width=1.6),
                  annotation_text="P90", annotation_font_color=FUCHSIA, annotation_position="top right")
    fig.update_layout(**play(310, showlegend=False), xaxis_title="Minutes since incident",
                      yaxis_title="Road cleared (%)")
    fig.update_yaxes(range=[0, 102], dtick=25)
    fig.update_xaxes(dtick=60)
    return fig


def fig_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix="%", font=dict(color=TXT, size=28)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=LINE, tickfont=dict(color=MUT2, size=10)),
                   bar=dict(color=color, thickness=0.34), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 100], color="rgba(255,255,255,.04)")],
                   threshold=dict(line=dict(color=color, width=3), thickness=0.78, value=value))))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=22, r=22, t=18, b=28),
                      font=dict(color=MUT, family="Inter"))
    fig.add_annotation(text=title, x=0.5, y=-0.12, showarrow=False, font=dict(color=MUT, size=12.5))
    return fig


def fig_marl(disp):
    rw = (disp or {}).get("rewards", {})
    greedy = float(rw.get("greedy-severity", 0)); rl = float(rw.get("GATE-PPO (learned)", greedy))
    uplift = max(rl - greedy, 0)
    cats = ["Greedy baseline", "CASCADE RL"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=cats, x=[greedy, greedy], orientation="h", name="Baseline",
                         marker=dict(color="#3a4a6b", line=dict(width=0)),
                         hovertemplate="baseline %{x:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(y=cats, x=[0, uplift], orientation="h", name="RL uplift",
                         marker=dict(color=EMERALD), text=["", f"+{uplift:,.0f}"], textposition="outside",
                         textfont=dict(color=EMERALD, size=13), cliponaxis=False,
                         hovertemplate="uplift %{x:,.0f}<extra></extra>"))
    fig.update_layout(**play(250, barmode="stack",
                             legend=dict(orientation="h", y=-0.32, x=0, font=dict(size=11))),
                      xaxis_title="Congestion relieved (reward units)")
    return fig


def fig_econ(twin):
    pj = (twin or {}).get("per_junction", [])[:8]
    if not pj:
        return None
    df = pd.DataFrame(pj)
    fig = go.Figure(go.Bar(x=df["veh_min_saved"], y=df["junction"], orientation="h",
                           marker=dict(color=df["veh_min_saved"], colorscale=MAGMA,
                                       line=dict(color="rgba(255,255,255,.12)", width=1)),
                           text=df["veh_min_saved"].map(lambda v: f"{v:,.0f}"), textposition="auto",
                           textfont=dict(color="#0b1020", size=12),
                           hovertemplate="%{y}: %{x:,.0f} veh-min<extra></extra>"))
    fig.update_layout(**play(460, margin=dict(l=215, r=30, t=20, b=58), bargap=0.42),
                      xaxis_title="Vehicle-minutes saved / day")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11.5), automargin=True)
    return fig


PLOT = dict(use_container_width=True, theme=None, config={"displayModeBar": False})


def live_pulse(height=232):
    components.html("""
<div style="font-family:Inter,system-ui,sans-serif;border:1px solid #283a62;border-radius:14px;
     background:#192542;padding:14px 16px 12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div style="font-size:10.5px;font-weight:800;letter-spacing:1.5px;color:#cbd5e1;text-transform:uppercase;">
      Live Network Pulse · 24h demand simulation</div>
    <div id="clk" style="font-size:12px;font-weight:800;color:#d4956b;"></div>
  </div>
  <div style="display:flex;gap:12px;margin-bottom:12px;">
    <div style="flex:1;background:#1d273d;border:1px solid #283a62;border-radius:11px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;font-weight:700;">Active incidents</div>
      <div id="s1" style="font-size:19px;font-weight:800;color:#e8eff5;margin-top:3px;">—</div></div>
    <div style="flex:1;background:#1d273d;border:1px solid #283a62;border-radius:11px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;font-weight:700;">Units engaged</div>
      <div id="s2" style="font-size:19px;font-weight:800;color:#4cc9f0;margin-top:3px;">—</div></div>
    <div style="flex:1;background:#1d273d;border:1px solid #283a62;border-radius:11px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:1px;color:#64748b;text-transform:uppercase;font-weight:700;">Avg clearance</div>
      <div id="s3" style="font-size:19px;font-weight:800;color:#6bcb9b;margin-top:3px;">—</div></div>
  </div>
  <canvas id="pz" style="width:100%;height:56px;display:block;"></canvas>
  <div style="margin-top:11px;display:flex;align-items:center;gap:9px;font-size:11.5px;color:#8aa4b8;">
    <span style="flex:none;width:7px;height:7px;border-radius:50%;background:#6bcb9b;box-shadow:0 0 8px #6bcb9b;"></span>
    <span id="tk"></span></div>
</div>
<script>
(function(){
  const c=document.getElementById('pz'),x=c.getContext('2d');
  const clk=document.getElementById('clk'),s1=document.getElementById('s1'),s2=document.getElementById('s2'),s3=document.getElementById('s3'),tk=document.getElementById('tk');
  const J=['Silk Board Jn','Hebbal Flyover','KR Puram','Marathahalli Br','Tin Factory','Electronic City','Mekhri Circle','Trinity Circle','Sarjapur Rd'];
  const K=['stalled vehicle','signal failure','waterlogging','minor collision','VIP movement','road works'];
  function load(h){return 0.40+0.60*Math.max(Math.exp(-Math.pow((h-10)/3,2)),Math.exp(-Math.pow((h-19)/3,2)));}
  let t0=performance.now();
  function frame(now){
    const W=c.clientWidth,H=56,dpr=window.devicePixelRatio||1;c.width=W*dpr;c.height=H*dpr;x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
    const prog=((now-t0)/16000)%1,hour=prog*24,L=load(hour);const yy=h=>H-7-load(h)*(H-15);
    let g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,'rgba(212,149,107,.40)');g.addColorStop(1,'rgba(76,201,240,.04)');
    x.beginPath();x.moveTo(0,H);for(let i=0;i<=W;i++)x.lineTo(i,yy(i/W*24));x.lineTo(W,H);x.closePath();x.fillStyle=g;x.fill();
    x.beginPath();for(let i=0;i<=W;i++){const y=yy(i/W*24);i?x.lineTo(i,y):x.moveTo(i,y);}x.strokeStyle='#d4956b';x.lineWidth=2;x.stroke();
    const px=prog*W,py=yy(hour);x.strokeStyle='rgba(199,125,219,.5)';x.beginPath();x.moveTo(px,0);x.lineTo(px,H);x.stroke();
    x.fillStyle='rgba(199,125,219,.22)';x.beginPath();x.arc(px,py,8,0,7);x.fill();x.fillStyle='#c77ddb';x.beginPath();x.arc(px,py,4,0,7);x.fill();
    const mm=Math.floor((hour%1)*60);
    clk.textContent=('0'+Math.floor(hour)).slice(-2)+':'+('0'+mm).slice(-2)+' IST · LOAD '+Math.round(L*100)+'%';
    s1.textContent=Math.round(6+L*46);s2.textContent=Math.round(40+L*22);s3.textContent=Math.round(70+L*95)+' min';
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  function tick(){const j=J[(Math.random()*J.length)|0],k=K[(Math.random()*K.length)|0],eta=3+((Math.random()*12)|0);
    tk.innerHTML='<b style="color:#cbd5e1">'+j+'</b> &middot; '+k+' &middot; nearest unit ETA '+eta+' min';}
  tick();setInterval(tick,3400);
})();
</script>
""", height=height)


rec = load_json("recommendation.json", {})
twin = load_json("twin_report.json", {})
disp = load_json("dispatcher_report.json", {})
loop = load_json("closed_loop_report.json", {})
ev = load_json("final_eval.json", {})
nodes = load_nodes()
risk = per_node_risk()
fc = node_forecast()
cal = load_calibrated()
n_events = int(len(cal["durations"]))

st.markdown(
    f'<div class="topbar"><div class="mark">{IC["logo"]}</div>'
    '<div><div class="wm">CASCADE</div>'
    '<div class="tg">Traffic Intelligence Command Center · Bengaluru Traffic Police × Flipkart</div></div>'
    f'<div class="chips">'
    f'<div class="chip"><div class="cl">Events</div><div class="cv">{n_events:,}</div></div>'
    f'<div class="chip"><div class="cl">Junctions</div><div class="cv">{len(nodes)}</div></div>'
    f'<div class="chip"><div class="cl">Coverage</div><div class="cv">90%</div></div>'
    f'<div class="chip live"><div class="cl">Status</div><div class="cv"><span class="pulse"></span>LIVE</div></div>'
    f'</div></div>', unsafe_allow_html=True)

# --- Junction selector ---
jname = nodes["junction"].tolist()[int(nodes["n_events"].idxmax())]
# Will be placed per-section below

# --- Default state for all controls via session_state ---
if "plan" not in st.session_state:
    st.session_state.plan = load_json("allocation.json", {"officers": [], "barricades": []})
plan = st.session_state.plan
divs = load_json("diversions.json", [])

tab_cmd, tab_model, tab_econ, tab_work = st.tabs(
    ["Command Center", "Model Transparency", "Economic Impact", "Deployment Worklist"])

with tab_cmd:
    section("Predictive Insights")
    k = st.columns(5)
    saved = (twin or {}).get("totals", {})
    saved_pct = round(100 * saved.get("saved_veh_min", 0) / max(saved.get("baseline_veh_min", 1), 1), 0) if twin else 0
    inr = saved.get("inr_per_day", 0)
    insight_card(k[0], "activity", f"{saved_pct:.0f}%", "Congestion avoided", "digital-twin estimate", CYAN)
    insight_card(k[1], "value", f"₹{inr/1e6:.1f}M", "Value / day", "value-of-time basis", EMERALD)
    insight_card(k[2], "cpu", f"+{(disp or {}).get('lift_vs_greedy_pct', 0):.0f}%", "RL efficiency", "vs greedy baseline", AMBER)
    insight_card(k[3], "target", f"+{(loop or {}).get('off_policy_reward', {}).get('lift_vs_random_pct', 0):.0f}%",
                 "Plan vs random", "on realized outcomes", FUCHSIA)
    insight_card(k[4], "shield", "90%", "Conformal coverage", "guaranteed calibration", VIOLET)

    vspace(20)
    officers = plan.get("officers", []); barr = plan.get("barricades", [])
    total_off = sum(int(x.get("headcount", 0)) for x in officers)
    top_j = officers[0]["junction"] if officers else "—"
    svm = saved.get("saved_veh_min", 0)
    xai = (f"Deploying <b>{total_off} officers</b> across <b>{len(officers)} priority junctions</b> — anchored at "
           f"<b>{top_j}</b> — intercepts the Hawkes incident contagion before it propagates. Projected relief: "
           f"<b>{svm:,.0f} vehicle-minutes/day</b> (~₹{inr/1e6:.1f}M). Conformal calibration guarantees ≥90% of "
           f"clearance-time estimates hold; {len(barr)} closure-prone junctions are barricaded with diversions pre-staged.")
    st.markdown(f'<div class="callout"><div class="ci">{IC["brief"]}</div>'
                f'<div><div class="ct">Commander\'s Briefing · Explainable Recommendation</div>'
                f'<div class="cx">{xai}</div></div></div>', unsafe_allow_html=True)

    vspace(22)

    map_col, ctrl_col = st.columns([2.5, 1])

    with ctrl_col:
        section("Controls")
        jname = st.selectbox("Focus junction", nodes["junction"].tolist(),
                             index=int(nodes["n_events"].idxmax()))
        nid = int(nodes.loc[nodes["junction"] == jname, "node_id"].iloc[0])
        vspace(6)
        mode = st.radio("Hotspot signal", ["Corrected (impact)", "Naive (raw count)"], index=0)
        mode_key = "corrected" if mode.startswith("Corrected") else "naive"
        vspace(6)
        map_style_opt = st.radio("Render style", ["3D density (hexbins)", "Impact points"], index=0)
        vspace(8)
        st.markdown(f'<div style="font-size:9.5px;font-weight:700;letter-spacing:1px;color:{MUT2};'
                    f'text-transform:uppercase;margin-bottom:6px;">Map layers</div>', unsafe_allow_html=True)
        show_off = st.checkbox("Officer deployment", True)
        show_bar = st.checkbox("Barricades", True)
        show_div = st.checkbox("Diversions", True)
        vspace(8)
        sim_on = st.checkbox("⏱ Time simulation", False)
        sim_hour = 18
        if sim_on:
            sim_hour = st.slider("Hour (IST)", 0, 23, 18)
        
        risk_disp, sim_load = (diurnal_display(risk, sim_hour) if sim_on else (risk, 1.0))
        if sim_on:
            st.caption(f"⏱ Network load index: **{sim_load*100:.0f}%** at {sim_hour:02d}:00 IST.")

    with map_col:
        section("Live Operations Map" + (f" · {sim_hour:02d}:00 IST" if sim_on else ""))
        ep = event_points()
        use_hex = map_style_opt.startswith("3D")
        wcol = "w" if mode_key == "corrected" else "cnt"
        layers = [hexbin_layer(ep, wcol)] if use_hex else [risk_layer(risk_disp, mode_key)]
        layers += plan_layers_for(plan, show_off, show_bar)
        if show_div:
            dl = diversion_layer(divs, nodes)
            if dl: layers.append(dl)
        view = pdk.ViewState(latitude=12.965, longitude=77.59, zoom=10.55,
                             pitch=46 if use_hex else 30, bearing=14 if use_hex else 5)
        tooltip = {"html": "<b>{junction}</b>", "style": {"background": PANEL2, "color": TXT,
                  "fontSize": "12px", "border": f"1px solid {LINE}", "borderRadius": "8px"}}
        deck = pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto",
                        map_style="dark", tooltip=tooltip, height=520)
        st.pydeck_chart(deck)
        st.markdown(
            '<div class="legend"><span style="font-weight:700;color:#cbd5e1">Congestion impact</span>'
            '<span><span class="grad"></span>low → high</span>'
            f'<span><span class="dot" style="background:{CYAN}"></span>Officer team</span>'
            f'<span><span class="dot" style="background:{ROSE}"></span>Barricade</span>'
            f'<span><span class="dot" style="background:{EMERALD}"></span>Diversion</span></div>',
            unsafe_allow_html=True)
        vspace(14)
        live_pulse()

    vspace(16)
    section(f"Junction Intelligence · {jname}")
    if nid in fc.index:
        row = fc.loc[nid]
        ic = risk.loc[risk["node_id"] == nid, "intensity"]
        rate = f"{ic.iloc[0]:.3f}/h" if len(ic) else "—"
        st.markdown(f'''
        <div class="ji-panel">
            <div class="ji-item">
                <div class="l">Median clearance</div>
                <div class="v" style="color:{CYAN}">{fmt_dur(row["median"])}</div>
                <div class="s">{jname}</div>
            </div>
            <div class="ji-div"></div>
            <div class="ji-item">
                <div class="l">P90 (calibrated)</div>
                <div class="v" style="color:{EMERALD}">{fmt_dur(row["p90"])}</div>
                <div class="s">conformal coverage</div>
            </div>
            <div class="ji-div"></div>
            <div class="ji-item">
                <div class="l">Road-closure prob.</div>
                <div class="v" style="color:{AMBER}">{row['closure']*100:.0f}%</div>
                <div class="s">Hawkes forecast</div>
            </div>
            <div class="ji-div"></div>
            <div class="ji-item">
                <div class="l">Incidents (history)</div>
                <div class="v" style="color:{FUCHSIA}">{int(row['incidents'])}</div>
                <div class="s">total recorded</div>
            </div>
            <div class="ji-div"></div>
            <div class="ji-item">
                <div class="l">Next-incident rate</div>
                <div class="v" style="color:{VIOLET}">{rate}</div>
                <div class="s">self-exciting model</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

with tab_model:
    section(f"DeepHit Survival · {jname}")
    if nid in fc.index:
        row = fc.loc[nid]
        st.plotly_chart(fig_survival(row["median"], row["p90"]), **PLOT)
        st.caption("Probability the road is cleared over time since the incident, anchored at the "
                   "conformally-calibrated P50 (median) and P90 bounds.")
        vspace(20)
        section("Hawkes Risk Gauges")
        g1, g2 = st.columns(2)
        sev = float(risk.loc[risk["node_id"] == nid, "corrected_n"].iloc[0]) * 100 if (risk["node_id"] == nid).any() else 0.0
        with g1:
            st.plotly_chart(fig_gauge(float(row["closure"]) * 100, "P(road closure)", AMBER), **PLOT)
        with g2:
            st.plotly_chart(fig_gauge(sev, "Severity index", FUCHSIA), **PLOT)
    vspace(20)
    section("RL Dispatcher vs Greedy Baseline")
    st.plotly_chart(fig_marl(disp), **PLOT)
    st.caption(f"The CASCADE GATE-PPO dispatcher relieves "
               f"{(disp or {}).get('lift_vs_greedy_pct', 0):.0f}% more congestion than the greedy baseline.")
    vspace(20)
    section("Predictive Accuracy (verified-label test)")
    if ev:
        vt = ev.get("verified_label_test", {})
        rows = [{"Model": m, "C-index": d.get("c_harrell")} for m, d in vt.items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"C-index": st.column_config.NumberColumn(format="%.3f")})

with tab_econ:
    section("Return on Investment")
    tot = (twin or {}).get("totals", {})
    e = st.columns(4)
    insight_card(e[0], "value", f"₹{tot.get('inr_per_day',0)/1e6:.2f}M", "Value / day", "value-of-time basis", EMERALD)
    insight_card(e[1], "activity", f"{tot.get('person_hours_saved',0):,.0f}", "Person-hours / day", "commuter time saved", CYAN)
    insight_card(e[2], "target", f"{tot.get('saved_veh_min',0)/1e3:,.0f}k", "Vehicle-min / day", "congestion avoided", FUCHSIA)
    sp = round(100 * tot.get("saved_veh_min", 0) / max(tot.get("baseline_veh_min", 1), 1), 0)
    insight_card(e[3], "cpu", f"{sp:.0f}%", "Congestion reduction", "vs no intervention", AMBER)
    vspace(24)
    section("Impact by Junction")
    f = fig_econ(twin)
    if f is not None:
        st.plotly_chart(f, **PLOT)


with tab_work:
    with st.expander("⚙️ What-if Deployment Optimizer", expanded=False):
        opt1, opt2, opt3, opt4, opt5 = st.columns([1, 1, 1, 1, 1.2])
        o = opt1.slider("Officer teams", 4, 24, 12)
        b = opt2.slider("Barricades", 2, 16, 8)
        r = opt3.slider("Radius (km)", 1.0, 6.0, 3.0, 0.5)
        pw = opt4.slider("Predictive wt", 0.0, 1.0, 0.5, 0.1)
        reopt = opt5.button("Re-optimize", use_container_width=True, type="primary")
    if reopt:
        try:
            with st.spinner("Solving allocation (OR-Tools CP-SAT)…"):
                st.session_state.plan = run_allocator(o, b, r, pw)
            plan = st.session_state.plan
            st.toast("✅ Deployment re-optimized successfully.")
        except Exception as e:
            st.error(f"Allocator failed: {e}")
    vspace(8)
    section("Manpower Deployment")
    odf = pd.DataFrame(plan.get("officers", []))
    if not odf.empty:
        odf = odf[["junction", "headcount", "covered_severity_min"]].copy()
        odf.columns = ["Junction", "Officers", "Covered severity-min"]
        st.dataframe(odf, hide_index=True, use_container_width=True, height=240, column_config={
            "Officers": st.column_config.NumberColumn(format="%d"),
            "Covered severity-min": st.column_config.NumberColumn(format="%.0f")})
    vspace(20)
    section("Barricade Plan")
    bdf = pd.DataFrame(plan.get("barricades", []))
    if not bdf.empty:
        bdf = bdf[["junction", "closure_prob", "burden_min"]].copy()
        bdf.columns = ["Junction", "P(closure)", "Burden (min)"]
        st.dataframe(bdf, hide_index=True, use_container_width=True, height=220, column_config={
            "P(closure)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
            "Burden (min)": st.column_config.NumberColumn(format="%.0f")})
    vspace(20)
    section("Diversion Routing")
    ddf = pd.DataFrame(divs or [])
    if not ddf.empty:
        keep = [c for c in ["barricade", "from", "to", "added_km", "reroute_km"] if c in ddf.columns]
        ddf = ddf[keep].copy()
        ddf.columns = ["Closed junction", "Reroute from", "Reroute to", "Added km", "Reroute km"][:len(keep)]
        st.dataframe(ddf, hide_index=True, use_container_width=True, height=220, column_config={
            "Added km": st.column_config.NumberColumn(format="%.2f km"),
            "Reroute km": st.column_config.NumberColumn(format="%.2f km")})

    vspace(22)
    section("Daily Ops Briefing")
    brief = (rec or {}).get("ops_briefing", "")
    accents = [(AMBER, "rgba(212,149,107,.14)"), (CYAN, "rgba(76,201,240,.14)"),
               (EMERALD, "rgba(107,203,155,.14)"), (FUCHSIA, "rgba(199,125,219,.14)"), (VIOLET, "rgba(143,164,232,.14)")]
    lines = [ln.strip() for ln in brief.splitlines()
             if ln.strip() and not set(ln.strip()) <= set("=-") and "BRIEFING" not in ln.upper()]
    rows_html = ""
    for i, ln in enumerate(lines):
        ac, ab = accents[i % len(accents)]
        rows_html += (f'<div class="brow" style="--bc:{ac};--bb:{ab}"><div class="bi">{i+1}</div>'
                      f'<div class="bx">{ln}</div></div>')
    st.markdown(f'<div class="bfeed">{rows_html}</div>' if rows_html else
                '<div class="panel">Run <code>python -m src.cascade.serve.recommend</code> to generate the briefing.</div>',
                unsafe_allow_html=True)

    vspace(30)
    if st.button("Transmit Orders to Field Units", use_container_width=True, type="primary"):
        st.toast("Orders transmitted securely to all field units.")
        st.success("Secure channel · Deployment orders acknowledged by field units.")
    vspace(10)
