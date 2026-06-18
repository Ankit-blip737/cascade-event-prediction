"""
final_eval.py — CASCADE Tier 0: the head-to-head the demo reports.

Compares the deep CASCADE trunk (GCN+GRU+DeepHit) against the XGBoost AFT baseline and their
rank-ensemble, on the held-out TEST set, sliced two ways:

  * all test events                  — the full set (heavily label-noised; see below)
  * verified-label test events       — only incidents with a REAL end timestamp
                                       (resolved/closed/end), i.e. duration is trustworthy.

Why the slice matters: only ~44% of events have a verified end time; the rest fall back to the
`modified_datetime` proxy, which corrupts `duration_min`. On the full noisy set every model is
pinned near C~0.60 (the labels, not the models, are the ceiling). On verified labels the deep
trunk's real advantage shows through.

Consumes the Colab outputs in models/ (preds_all.npz) and refits the XGBoost AFT baseline locally
(deterministic) so the table is reproducible end-to-end with one command.

Usage:
    python -m src.cascade.eval.final_eval
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger("cascade.final_eval")


def _setup_logging(verbose=True):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


def _xgb_aft_risk(fb, num, cat, split, dur, ev):
    """Refit the XGBoost AFT baseline on the train split; return -predicted_time (risk) for all rows."""
    import xgboost as xgb

    def X(df):
        Z = df[num].astype(float).copy()
        for c in cat:
            Z[c] = df[c].astype("category")
        return Z

    def bounds(t, e):
        lo = np.clip(t, 1, None).astype(float)
        return lo, np.where(e == 1, lo, np.inf)

    tr, va = split == 0, split == 1
    lo_tr, hi_tr = bounds(dur[tr], ev[tr]); lo_va, hi_va = bounds(dur[va], ev[va])
    dtr = xgb.DMatrix(X(fb[tr]), enable_categorical=True)
    dtr.set_float_info("label_lower_bound", lo_tr); dtr.set_float_info("label_upper_bound", hi_tr)
    dva = xgb.DMatrix(X(fb[va]), enable_categorical=True)
    dva.set_float_info("label_lower_bound", lo_va); dva.set_float_info("label_upper_bound", hi_va)
    dall = xgb.DMatrix(X(fb), enable_categorical=True)
    params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
              "tree_method": "hist", "learning_rate": 0.05, "max_depth": 4,
              "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 7}
    bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dva, "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    return -bst.predict(dall)


def run(preds_path, features_path, meta_path, events_path, outdir, ens_w=0.4):
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score

    p = np.load(preds_path, allow_pickle=True)
    pmf, cuts = p["pmf"], p["cuts"]
    split, dur, ev = p["split"], p["durations"], p["events"]
    eid = p["event_id"].astype(str)
    K = pmf.shape[1]
    risk_dl = -(pmf * np.arange(K)).sum(1)   # outlier-immune expected-bin risk (matches notebook)

    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    num, cat = meta["numeric_features"], meta["categorical_features"]
    feats = pd.read_parquet(features_path).set_index("id").loc[eid].reset_index()
    risk_xgb = _xgb_aft_risk(feats, num, cat, split, dur, ev)
    risk_ens = ens_w * rankdata(risk_dl) + (1 - ens_w) * rankdata(risk_xgb)

    # verified-label mask: a real end timestamp existed (not the modified_datetime proxy)
    raw = pd.read_parquet(events_path).set_index("id")
    rel = (raw["resolved_datetime"].notna() | raw["closed_datetime"].notna() | raw["end_datetime"].notna())
    reliable = rel.reindex(eid).fillna(False).to_numpy()

    te = split == 2
    slices = {"all_test": te, "verified_label_test": te & reliable}
    tr = split == 0
    y_tr = Surv.from_arrays(ev[tr].astype(bool), np.clip(dur[tr], 1, None))

    models = {"GNN deep (CASCADE)": risk_dl, "XGBoost AFT (baseline)": risk_xgb,
              f"Ensemble {int(ens_w*100)}/{int((1-ens_w)*100)}": risk_ens}

    results = {}
    for sname, mask in slices.items():
        y_s = Surv.from_arrays(ev[mask].astype(bool), np.clip(dur[mask], 1, None))
        tau = float(np.quantile(dur[mask][ev[mask].astype(bool)], 0.95))
        results[sname] = {}
        for mname, r in models.items():
            cH = concordance_index_censored(ev[mask].astype(bool), dur[mask], r[mask])[0]
            try:
                cI = concordance_index_ipcw(y_tr, y_s, r[mask], tau=tau)[0]
            except Exception:
                cI = float("nan")
            results[sname][mname] = {"c_harrell": round(float(cH), 4), "c_ipcw": round(float(cI), 4),
                                     "n": int(mask.sum())}

    _report(results)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "final_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved -> %s", outdir / "final_eval.json")
    return results


def _report(results):
    logger.info("=" * 78)
    logger.info("FINAL EVALUATION  (C-index; higher is better)")
    for sname, d in results.items():
        n = next(iter(d.values()))["n"]
        logger.info("  -- %s (n=%d) --", sname, n)
        logger.info("     %-26s %10s %10s", "model", "C-Harrell", "C-IPCW")
        for mname, m in d.items():
            logger.info("     %-26s %10.4f %10.4f", mname, m["c_harrell"], m["c_ipcw"])
    logger.info("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Final GNN-vs-baseline-vs-ensemble comparison.")
    ap.add_argument("--preds", default="models/preds_all.npz")
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--meta", default="data/processed/feature_meta.json")
    ap.add_argument("--events", default="data/processed/events_clean.parquet")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--ens-w", type=float, default=0.4, help="GNN weight in the rank ensemble")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _setup_logging(not args.quiet)
    for pth in (args.preds, args.features, args.meta, args.events):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Missing input: {pth}")
    run(args.preds, args.features, args.meta, args.events, args.outdir, args.ens_w)


if __name__ == "__main__":
    main()
