"""
closed_loop.py — CASCADE Tier 3: the learning loop (de-censoring + intensity residual + off-policy reward).

Closes the deploy -> observe -> learn cycle by treating the held-out TEST period as "what actually
happened next" and turning it into three concrete update signals:

  1. DE-CENSORING — incidents that were still active (censored) at prediction time but resolve later
     give newly-observed durations; we count them and re-run Adaptive Conformal Inference so the
     coverage guarantee tracks the drift (split conformal alone is void under retraining/censoring).
  2. INTENSITY RESIDUAL — the Hawkes head predicted a per-junction arrival rate; we compare its share
     of predicted arrivals against the actual share observed in the test period. Large positive
     residuals are model blind spots (under-predicted hotspots) -> the correction signal.
  3. OFF-POLICY REWARD — score the deployed plan against realized outcomes: how much true congestion
     the staffed junctions actually carried vs random placement -> did the deployment pay off.

Pure local orchestration over the saved artifacts. Output: models/closed_loop_report.json.

Usage:
    python -m src.cascade.closed_loop
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.cascade.calibrate.aci import ACI
from src.cascade.calibrate.conformal_survival import predict_quantile

logger = logging.getLogger("cascade.closed_loop")
CAP = 1440.0


def _setup_logging(v=True):
    logging.basicConfig(level=logging.INFO if v else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")


def de_censor(p, alpha=0.1, gamma=0.02):
    """Count newly-resolved (previously censored) incidents and re-run ACI over the test stream."""
    split, ev, dur, pmf, cuts = p["split"], p["events"], p["durations"], p["pmf"], p["cuts"]
    cal, te = split == 1, split == 2
    newly_resolved = int(((te) & (ev == 1)).sum())     # observed outcomes that update labels/bounds
    still_active = int(((te) & (ev == 0)).sum())

    qhat_cal = predict_quantile(pmf[cal], cuts, 1 - alpha)
    cal_scores = dur[cal][ev[cal].astype(bool)] - qhat_cal[ev[cal].astype(bool)]
    te_idx = np.where(te & (ev == 1))[0]
    qhat_te = predict_quantile(pmf[te_idx], cuts, 1 - alpha)
    aci = ACI(alpha, gamma); covered = 0
    for i in range(len(te_idx)):
        bound = qhat_te[i] + aci.offset(cal_scores, aci.alpha)
        c = dur[te_idx[i]] <= bound; covered += int(c); aci.update(c)
    return {"newly_resolved": newly_resolved, "still_active_censored": still_active,
            "aci_alpha_drift": [round(alpha, 3), round(aci.alpha, 3)],
            "recalibrated_coverage": round(covered / max(len(te_idx), 1), 3)}


def intensity_residual(p, top=8):
    """Predicted vs actual per-junction arrival share over the test period -> blind-spot residuals."""
    split, node_id = p["split"], p["node_id"]
    node_int = p["node_intensity"]
    V = len(node_int)
    te = split == 2
    actual = np.bincount(node_id[te], minlength=V).astype(float)
    actual_share = actual / max(actual.sum(), 1)
    pred_share = node_int / max(node_int.sum(), 1e-9)
    resid = actual_share - pred_share
    order = np.argsort(-resid)[:top]
    under = [{"node_id": int(j), "actual_share": round(float(actual_share[j]), 4),
              "pred_share": round(float(pred_share[j]), 4), "residual": round(float(resid[j]), 4)}
             for j in order if resid[j] > 0]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {"share_rmse": round(rmse, 5), "top_under_predicted": under}


def off_policy_reward(p, allocation_path, response_gain=0.3, seed=0):
    """Realized congestion the staffed junctions carried vs random placement (counterfactual)."""
    split, node_id, dur = p["split"], p["node_id"], p["durations"]
    road, ev = p["road_closure"], p["events"]
    V = len(p["node_intensity"]); te = split == 2
    realized = np.zeros(V)                                  # true congestion-min per node in test
    sev = np.clip(dur[te], None, CAP) * road[te] * ev[te]
    np.add.at(realized, node_id[te], sev)

    plan = json.loads(Path(allocation_path).read_text(encoding="utf-8"))
    staffed = [int(o["node_id"]) for o in plan.get("officers", [])]
    deployed = response_gain * realized[staffed].sum()

    rng = np.random.default_rng(seed)
    rand = np.array([response_gain * realized[rng.choice(V, len(staffed), replace=False)].sum()
                     for _ in range(200)]).mean()
    lift = 100 * (deployed - rand) / max(rand, 1e-9)
    return {"deployed_relieved": round(float(deployed), 0), "random_relieved": round(float(rand), 0),
            "lift_vs_random_pct": round(float(lift), 1), "n_staffed": len(staffed)}


def run(preds_path, allocation_path, out_path):
    p = np.load(preds_path, allow_pickle=True)
    dc = de_censor(p)
    ir = intensity_residual(p)
    op = off_policy_reward(p, allocation_path)

    logger.info("=" * 64)
    logger.info("CLOSED-LOOP UPDATE  (test period treated as newly-observed reality)")
    logger.info("  de-censoring: %d newly resolved, %d still active", dc["newly_resolved"], dc["still_active_censored"])
    logger.info("     ACI alpha %s -> recalibrated coverage %.1f%%", dc["aci_alpha_drift"], 100 * dc["recalibrated_coverage"])
    logger.info("  intensity residual: share RMSE %.4f; top blind spots:", ir["share_rmse"])
    for u in ir["top_under_predicted"][:5]:
        logger.info("     node %d  actual %.3f vs pred %.3f  (residual +%.3f)",
                    u["node_id"], u["actual_share"], u["pred_share"], u["residual"])
    logger.info("  off-policy reward: deployed %.0f vs random %.0f  -> %+.1f%% (n=%d staffed)",
                op["deployed_relieved"], op["random_relieved"], op["lift_vs_random_pct"], op["n_staffed"])
    logger.info("=" * 64)

    report = {"de_censoring": dc, "intensity_residual": ir, "off_policy_reward": op}
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved closed-loop report -> %s", out_path)
    return report


def main():
    ap = argparse.ArgumentParser(description="Closed-loop learning signals from realized outcomes.")
    ap.add_argument("--preds", default="models/preds_mtl.npz")
    ap.add_argument("--allocation", default="models/allocation.json")
    ap.add_argument("--out", default="models/closed_loop_report.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    for pth in (args.preds, args.allocation):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Missing {pth}")
    run(args.preds, args.allocation, args.out)


if __name__ == "__main__":
    main()
