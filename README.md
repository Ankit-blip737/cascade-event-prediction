# 🚦 CASCADE — Bengaluru Traffic Intelligence

**Event-driven congestion forecasting & resource dispatch for the Bengaluru Traffic Police.**
Built for the Flipkart × BTP **Gridlock Hackathon 2.0** (Round 2) on the provided ASTRAM incident log.

CASCADE turns a raw incident-response log into an operational decision system: it **de-biases** the
hotspot map, **forecasts** how long each incident will block the road (with a *coverage guarantee*),
predicts **where the next incident will fire**, and recommends the optimal **manpower, barricading and
diversion** plan — then validates the plan against realized outcomes in a closed loop.

---

## The core insight
A raw incident heatmap shows **where patrols logged tickets**, not where congestion actually
concentrates. CASCADE separates **exposure** (counts) from **impact** (severity × duration × closure ×
contagion). The naive→corrected map toggle is the hero of the demo — the gap between them is the
project.

---

## 🖥️ The dashboard (what judges see)
A single Streamlit app: de-biased hotspot map (naive vs corrected), per-junction calibrated forecasts,
the deployment plan (officers / barricades / diversions) on a GPU map, a **live what-if re-optimizer**,
and the measured impact + daily ops briefing.

```bash
# from the repo root
pip install -r requirements-app.txt
streamlit run src/cascade/demo/app.py
# open http://localhost:8501
```

### Deploy (one public URL)
- **Streamlit Community Cloud / Hugging Face Spaces:** point at `src/cascade/demo/app.py`, set
  `requirements-app.txt` as the requirements file. (Repo already ships the small artifacts it reads.)
- **Docker (Render / Railway / Fly):** `docker build -t cascade . && docker run -p 8501:8501 cascade`.
- *(Optional)* set `MAPPLS_KEY=...` to enable Mappls (MapmyIndia) basemap + real-route diversions;
  without it the app uses the free CARTO basemap + offline routing (auto-fallback, never breaks).

---

## 🏗️ Architecture (tiered — all built & validated)

```
RAW ASTRAM CSV
  → ETL: ingest (survival labels) · junction graph (294 nodes) · features (causal ripple) · bundle
  → SHARED TRUNK  GCN (spatial) + GRU (temporal)        [Mamba selective-SSM swap available]
  → MULTI-TASK HEADS  (MMoE + GradNorm→PCGrad)
        • DeepHit survival  → duration distribution + P(road-closure) + priority
        • Hawkes intensity  → where/when the next incident fires
  → CALIBRATION  Conformal + Adaptive Conformal Inference   (provable coverage under drift)
  → DECISION  OR-Tools allocator (officers + barricades + diversions, predictive)
             · GATE-PPO RL dispatcher · SPO+ decision-focused tuning
  → DIGITAL TWIN  deterministic-queue impact + ₹/day economics
  → CLOSED LOOP  de-censoring · intensity residual · off-policy reward
  → SERVE  unified recommendation + grounded ops briefing  →  Dashboard
```

## 📊 Key results (held-out test)
| Metric | Result |
|---|---|
| Survival C-index (verified-label incidents) | **0.667** (GNN) — beats XGBoost 0.663; ensemble 0.670 |
| Calibration (Integrated Brier) | multi-task **0.0514** < XGBoost AFT 0.0546 |
| Conformal coverage (target 90%) | **90.7%** on held-out test (+ ACI holds 90.0% under drift) |
| Hawkes next-incident C-index | **0.645** (no baseline can do this) |
| RL dispatcher vs greedy | **+20.3%** congestion relieved |
| Plan vs random (realized outcomes) | **+360%** |
| Digital-twin impact (illustrative) | ~51% congestion avoided · ~₹4.9M/day |

> Honest notes: only ~44% of incidents have a verified end-time (the rest use a `modified_datetime`
> proxy), so we report C-index on both all-test and verified-label slices. SPO+ ties the two-stage
> baseline (a legitimate decision-focused-learning finding). Twin economics are illustrative; the
> assumptions live in `models/twin_report.json`.

---

## 🔁 Reproduce from the raw data
```bash
pip install -r requirements.txt        # full (training) deps
python -m src.cascade.data.ingest   --input data/raw/astram_events.csv --output data/processed/events_clean.parquet
python -m src.cascade.data.graph
python -m src.cascade.data.features
python -m src.cascade.data.dataset                       # -> train_bundle.npz (upload to Colab)
python -m src.cascade.eval.baselines                     # XGBoost baseline
#   Colab (GPU): run notebooks/02_hawkes_mmoe.ipynb -> download trunk_mtl.pt + preds_mtl.npz to models/
python -m src.cascade.eval.final_eval
python -m src.cascade.calibrate.conformal_survival --preds models/preds_mtl.npz --out models/calibrated.npz
python -m src.cascade.optimize.allocator --scope test --predict-weight 0.5
python -m src.cascade.optimize.diversion
python -m src.cascade.twin.sumo_runner
python -m src.cascade.rl.gate_ppo --updates 100          # trains locally on CPU
python -m src.cascade.diffusion.ddpm
python -m src.cascade.closed_loop
python -m src.cascade.serve.recommend                    # -> models/recommendation.json (UI reads this)
```

## 📁 Repo layout
```
src/cascade/
  data/      ingest · graph · features · dataset
  trunk/     mamba_block (selective-SSM swap; GRU is the default in the notebooks)
  eval/      metrics · baselines · final_eval
  calibrate/ conformal_survival · aci
  optimize/  allocator · diversion
  twin/      sumo_runner (analytical queue + economics)
  rl/        dispatch_env · gate_ppo (GATE-PPO dispatcher)
  diffusion/ ddpm (rare planned-event augmentation)
  geo/       mappls (optional, guarded, auto-fallback)
  serve/     recommend (unified payload + ops briefing)
  demo/      app.py  ← the dashboard
notebooks/   01 trunk+DeepHit · 02 Hawkes+MMoE+PCGrad · 03 SPO+   (Colab GPU)
models/      trained weights + all computed JSON/npz artifacts
data/        raw + processed
HANDOFF.md   detailed build/handoff notes
```

Built only on the provided ASTRAM dataset. Mappls/MapmyIndia is used as an allowed *service* (maps /
snap-to-road / routing), never as a modeling dataset.
