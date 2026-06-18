"""
allocator.py — CASCADE Tier 0, Component #12: the decision layer (OR-Tools).

Turns calibrated per-incident predictions into a concrete deployment plan:
  * OFFICERS  — a Maximal Coverage Location problem (CP-SAT): place P teams on junctions to cover
                the most "severity-minutes" of expected congestion within a response radius.
  * BARRICADES — a budgeted selection ILP (CP-SAT): pick B junctions maximizing closure-driven
                 spillover (P(road_closure) x traffic burden), with optional spatial separation.

Demand model (per junction): aggregate the conformally-calibrated P90 clearance time (upper bound,
the worst-case officer-minutes of work) over the incidents in scope, then weight by predicted
road-closure probability for barricade spillover. Because the duration input is the *coverage-
guaranteed* upper bound, the plan is robust, not point-estimate-optimistic.

Inputs: models/calibrated.npz (from conformal_survival.py) + data/processed/graph_nodes.parquet.
Output: models/allocation.json — officer coordinates + headcount, ranked barricade junctions.

Usage:
    python -m src.cascade.optimize.allocator \
        --calibrated models/calibrated.npz --nodes data/processed/graph_nodes.parquet \
        --officers 12 --barricades 8 --radius-km 3.0 --manpower 60
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

# Import OR-Tools BEFORE numpy/pandas: pyarrow (pulled in by pandas.read_parquet) and CP-SAT ship
# conflicting runtimes, and loading pyarrow first makes the solver segfault on this platform.
from ortools.sat.python import cp_model

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.allocator")

EARTH_RADIUS_KM = 6371.0088


def _setup_logging(verbose=True):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


def _haversine_matrix(lat, lon):
    la = np.radians(lat)[:, None]; lo = np.radians(lon)[:, None]
    dlat = la - la.T; dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_demand(cal, nodes, scope, severity_cap_min=1440.0, predict_weight=0.0, horizon_h=24.0):
    """Aggregate calibrated incident predictions to per-junction demand for the chosen scope.

    REACTIVE demand = sum of calibrated P90 clearance minutes (capped at 24h) over scope incidents.
    PREDICTIVE demand (Tier 1) = the Hawkes head's per-junction next-incident rate x horizon x a
    typical per-incident severity -> expected future severity-minutes. `predict_weight` blends them,
    so officers/barricades can be PRE-POSITIONED where the next incidents are forecast, not only
    where load is high right now."""
    split = cal["split"]
    sel = np.ones(len(split), bool) if scope == "all" else (split == {"train": 0, "val": 1, "test": 2}[scope])
    df = pd.DataFrame({
        "node_id": cal["node_id"][sel],
        "upper": np.minimum(cal["upper"][sel], severity_cap_min),   # calibrated P90 clearance, capped
        "closure": cal["closure_prob"][sel],
        "priority": cal["priority_prob"][sel],
    })
    g = df.groupby("node_id").agg(reactive_min=("upper", "sum"), n=("upper", "size"),
                                  closure=("closure", "mean"), priority=("priority", "mean"))
    nd = nodes.set_index("node_id").join(g, how="left").fillna({"reactive_min": 0, "n": 0,
                                                                "closure": 0, "priority": 0})

    per_inc = float(np.median(df["upper"])) if len(df) else 0.0     # typical incident severity-minutes
    if predict_weight > 0 and "node_intensity" in cal.files:
        inten = cal["node_intensity"]
        nd["intensity_h"] = [float(inten[i]) if i < len(inten) else 0.0 for i in nd.index]
        nd["predictive_min"] = nd["intensity_h"] * horizon_h * per_inc
        nd["load_min"] = (1 - predict_weight) * nd["reactive_min"] + predict_weight * nd["predictive_min"]
    else:
        nd["intensity_h"] = 0.0
        nd["predictive_min"] = 0.0
        nd["load_min"] = nd["reactive_min"]

    nd["barricade_score"] = nd["closure"] * nd["load_min"]   # spillover: closure prob x burden
    return nd.reset_index()


def place_officers(nd, n_officers, radius_km, manpower, time_limit=10.0):
    """Maximal Coverage Location (CP-SAT): P teams covering the most severity-minutes within R km."""
    active = nd[nd["load_min"] > 0].reset_index(drop=True)
    if active.empty:
        return [], 0.0, float(nd["load_min"].sum())
    lat, lon = active["lat"].to_numpy(), active["lon"].to_numpy()
    demand = np.rint(active["load_min"].to_numpy()).astype(int)
    D = _haversine_matrix(lat, lon)
    n = len(active)

    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{j}") for j in range(n)]      # facility (officer team) at junction j
    x = [m.NewBoolVar(f"x{i}") for i in range(n)]      # demand i covered
    m.Add(sum(y) <= n_officers)
    cover_sets = []
    for i in range(n):
        covering = [y[j] for j in range(n) if D[i, j] <= radius_km]
        cover_sets.append([j for j in range(n) if D[i, j] <= radius_km])
        m.Add(x[i] <= sum(covering)) if covering else m.Add(x[i] == 0)
    m.Maximize(sum(int(demand[i]) * x[i] for i in range(n)))

    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = time_limit
    s.Solve(m)

    chosen = [j for j in range(n) if s.Value(y[j]) > 0.5]
    covered = [i for i in range(n) if s.Value(x[i]) > 0.5]
    total_demand = float(active["load_min"].sum())
    covered_demand = float(active.loc[covered, "load_min"].sum())

    # attribute each covered junction to its nearest chosen team -> headcount split
    per_team = {j: 0.0 for j in chosen}
    for i in covered:
        nearest = min(chosen, key=lambda j: D[i, j])
        per_team[nearest] += active.loc[i, "load_min"]
    plan = []
    for j in chosen:
        share = per_team[j] / max(covered_demand, 1e-9)
        plan.append({
            "node_id": int(active.loc[j, "node_id"]), "junction": active.loc[j, "junction"],
            "lat": float(active.loc[j, "lat"]), "lon": float(active.loc[j, "lon"]),
            "covered_severity_min": round(per_team[j], 1),
            "headcount": int(max(1, round(manpower * share))),
        })
    plan.sort(key=lambda d: -d["covered_severity_min"])
    return plan, covered_demand, total_demand


def place_barricades(nd, n_barricades, min_sep_km, time_limit=10.0):
    """Budgeted ILP (CP-SAT): pick B junctions maximizing closure-spillover, optionally spatially spread."""
    cand = nd[nd["barricade_score"] > 0].reset_index(drop=True)
    if cand.empty:
        return []
    score = np.rint(cand["barricade_score"].to_numpy()).astype(int)
    lat, lon = cand["lat"].to_numpy(), cand["lon"].to_numpy()
    n = len(cand)

    m = cp_model.CpModel()
    b = [m.NewBoolVar(f"b{k}") for k in range(n)]
    m.Add(sum(b) <= n_barricades)
    if min_sep_km > 0:                                  # don't cluster barricades on top of each other
        D = _haversine_matrix(lat, lon)
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < min_sep_km:
                    m.Add(b[i] + b[j] <= 1)
    m.Maximize(sum(int(score[k]) * b[k] for k in range(n)))

    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = time_limit
    s.Solve(m)
    chosen = [k for k in range(n) if s.Value(b[k]) > 0.5]
    out = [{
        "node_id": int(cand.loc[k, "node_id"]), "junction": cand.loc[k, "junction"],
        "lat": float(cand.loc[k, "lat"]), "lon": float(cand.loc[k, "lon"]),
        "closure_prob": round(float(cand.loc[k, "closure"]), 3),
        "burden_min": round(float(cand.loc[k, "load_min"]), 1),
        "spillover_score": round(float(cand.loc[k, "barricade_score"]), 1),
    } for k in chosen]
    out.sort(key=lambda d: -d["spillover_score"])
    return out


def run(calibrated_path, nodes_path, n_officers, n_barricades, radius_km, manpower, min_sep_km, scope,
        out_path, predict_weight=0.0, horizon_h=24.0):
    cal = np.load(calibrated_path, allow_pickle=True)
    nodes = pd.read_parquet(nodes_path)
    nd = build_demand(cal, nodes, scope, predict_weight=predict_weight, horizon_h=horizon_h)
    has_pred = predict_weight > 0 and "node_intensity" in cal.files

    officers, cov, tot = place_officers(nd, n_officers, radius_km, manpower)
    barricades = place_barricades(nd, n_barricades, min_sep_km)

    logger.info("=" * 70)
    mode = ("REACTIVE+PREDICTIVE (w=%.2f, %.0fh horizon)" % (predict_weight, horizon_h)) if has_pred else "REACTIVE"
    logger.info("DEPLOYMENT PLAN  (scope=%s, mode=%s, %d active junctions)",
                scope, mode, int((nd["load_min"] > 0).sum()))
    logger.info("  OFFICERS: %d teams, %d total manpower", len(officers), sum(o["headcount"] for o in officers))
    logger.info("    severity-minutes covered: %.0f / %.0f  (%.1f%% within %.1f km)",
                cov, tot, 100 * cov / max(tot, 1e-9), radius_km)
    for o in officers[:6]:
        logger.info("    team -> %-22s (%.4f,%.4f)  %2d officers  cover=%.0f min",
                    o["junction"], o["lat"], o["lon"], o["headcount"], o["covered_severity_min"])
    logger.info("  BARRICADES: %d junctions (closure-prob x burden)", len(barricades))
    for bnode in barricades[:6]:
        logger.info("    barricade -> %-22s p(close)=%.2f  burden=%.0f min  score=%.0f",
                    bnode["junction"], bnode["closure_prob"], bnode["burden_min"], bnode["spillover_score"])
    logger.info("=" * 70)

    out = {
        "params": {"officers": n_officers, "barricades": n_barricades, "radius_km": radius_km,
                   "manpower": manpower, "min_sep_km": min_sep_km, "scope": scope,
                   "predict_weight": predict_weight, "horizon_h": horizon_h, "mode": mode},
        "coverage": {"covered_severity_min": round(cov, 1), "total_severity_min": round(tot, 1),
                     "covered_fraction": round(cov / max(tot, 1e-9), 4)},
        "officers": officers, "barricades": barricades,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Saved deployment plan -> %s", out_path)
    return out


def main():
    ap = argparse.ArgumentParser(description="OR-Tools deployment allocator.")
    ap.add_argument("--calibrated", default="models/calibrated.npz")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.parquet")
    ap.add_argument("--officers", type=int, default=12)
    ap.add_argument("--barricades", type=int, default=8)
    ap.add_argument("--radius-km", type=float, default=3.0)
    ap.add_argument("--manpower", type=int, default=60, help="total officers to split across teams")
    ap.add_argument("--min-sep-km", type=float, default=0.5, help="min spacing between barricades")
    ap.add_argument("--scope", choices=["all", "train", "val", "test"], default="test")
    ap.add_argument("--predict-weight", type=float, default=0.0,
                    help="0=reactive only; 0.5 blends Tier-1 predicted next-incident intensity")
    ap.add_argument("--horizon-h", type=float, default=24.0, help="lookahead window for predictive demand")
    ap.add_argument("--out", default="models/allocation.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    for pth in (args.calibrated, args.nodes):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Missing {pth} (run conformal_survival.py + graph.py first)")
    run(args.calibrated, args.nodes, args.officers, args.barricades, args.radius_km,
        args.manpower, args.min_sep_km, args.scope, args.out,
        predict_weight=args.predict_weight, horizon_h=args.horizon_h)


if __name__ == "__main__":
    main()
