"""
baselines.py — CASCADE Tier 0, Component #6: the survival baseline to beat.

Trains classical survival models on the feature matrix and scores them with the censoring-aware
metrics in metrics.py. This produces the FIRST NUMBER the GCN+GRU+DeepHit model must beat.

Models:
  * Kaplan-Meier   marginal survival, no covariates — the naive IBS reference.
  * Cox PH         (lifelines) semi-parametric linear hazard — classic survival baseline.
  * XGBoost AFT    (objective=survival:aft) censoring-native gradient boosting — the strong baseline.
  * XGBoost Cox    (objective=survival:cox) risk-ranking variant — best-case concordance reference.

Light enough to run LOCALLY on CPU in seconds (8k rows); only the deep heads go to Colab.

Split: the chronological train/val/test from features.py. Trains on train(+val for XGB early stop),
evaluates on the held-out test set. Writes a metrics table + the fitted XGBoost AFT model + test
predictions to models/baselines/.

Usage:
    python -m src.cascade.eval.baselines \
        --features data/processed/features.parquet \
        --meta     data/processed/feature_meta.json \
        --outdir   models/baselines
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.cascade.eval import metrics as M

logger = logging.getLogger("cascade.baselines")

DURATION_COL = "duration_min"
EVENT_COL = "event_observed"
MIN_DURATION = 1.0  # clip floor (minutes) so AFT log-bounds and Cox are well-defined


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --- data prep ----------------------------------------------------------------
def load_split(features_path, meta_path):
    feats = pd.read_parquet(features_path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    num = meta["numeric_features"]
    cat = meta["categorical_features"]

    feats = feats[feats["split"].isin(["train", "val", "test"])].copy()
    feats[DURATION_COL] = feats[DURATION_COL].clip(lower=MIN_DURATION)

    def _slice(which):
        d = feats[feats["split"] == which]
        return d, d[EVENT_COL].to_numpy().astype(int), d[DURATION_COL].to_numpy(float)

    (tr, e_tr, t_tr) = _slice("train")
    (va, e_va, t_va) = _slice("val")
    (te, e_te, t_te) = _slice("test")
    logger.info("Split sizes - train %d (%.1f%% obs), val %d, test %d (%.1f%% obs)",
                len(tr), 100 * e_tr.mean(), len(va), len(te), 100 * e_te.mean())
    return feats, meta, num, cat, (tr, e_tr, t_tr), (va, e_va, t_va), (te, e_te, t_te)


def _xy_xgb(df, num, cat):
    """Feature frame for XGBoost: numerics as float + category codes as pandas 'category' dtype."""
    X = df[num].astype(float).copy()
    for c in cat:
        X[c] = df[c].astype("category")
    return X


def _x_cox(df, num, mu=None, sd=None, keep=None):
    """Standardised numeric matrix for Cox (z-scored on TRAIN stats; codes excluded — freq encodings
    already carry category signal). The kept-column set is decided ONCE on train and reused verbatim
    on val/test, so the design matrices always align."""
    X = df[num].astype(float).copy()
    fit_mode = mu is None
    if fit_mode:
        mu, sd = X.mean(), X.std().replace(0, 1.0)
    Xz = (X - mu) / sd
    if keep is None:  # train: pick non-degenerate columns
        keep = [c for c in Xz.columns if np.isfinite(Xz[c]).all() and Xz[c].nunique() > 1]
    return Xz[keep], mu, sd, keep


# --- models -------------------------------------------------------------------
def fit_km(t_tr, e_tr, grid):
    """Kaplan-Meier marginal survival broadcast to every test row (naive IBS reference)."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(t_tr, e_tr)
    sf = km.survival_function_at_times(grid).to_numpy()  # [n_times]
    return sf


def fit_cox(tr, te, num, t_tr, e_tr, grid):
    """lifelines Cox PH. Returns (risk_test, surv_prob_test[n_test,n_times]) or (None,None)."""
    from lifelines import CoxPHFitter

    Xz_tr, mu, sd, keep = _x_cox(tr, num)
    Xz_te, *_ = _x_cox(te, num, mu, sd, keep=keep)

    df_tr = Xz_tr.copy()
    df_tr["_t"] = np.clip(t_tr, MIN_DURATION, None)
    df_tr["_e"] = e_tr
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_tr, duration_col="_t", event_col="_e", show_progress=False)

    risk = cph.predict_partial_hazard(Xz_te).to_numpy().ravel()  # higher => higher risk
    sf = cph.predict_survival_function(Xz_te, times=grid)         # [n_times, n_test]
    surv = sf.to_numpy().T                                        # [n_test, n_times]
    return risk, surv


def fit_xgb_aft(Xtr, Xva, Xte, t_tr, e_tr, t_va, e_va, grid,
                dist="normal", scale=1.2, rounds=600):
    """XGBoost AFT. Returns (risk_test, surv_prob_test, booster)."""
    import xgboost as xgb
    from scipy.stats import norm

    def _bounds(t, e):
        lo = np.clip(t, MIN_DURATION, None).astype(float)
        hi = np.where(e == 1, lo, np.inf)  # censored => upper bound = +inf
        return lo, hi

    lo_tr, hi_tr = _bounds(t_tr, e_tr)
    lo_va, hi_va = _bounds(t_va, e_va)

    dtr = xgb.DMatrix(Xtr, enable_categorical=True)
    dtr.set_float_info("label_lower_bound", lo_tr)
    dtr.set_float_info("label_upper_bound", hi_tr)
    dva = xgb.DMatrix(Xva, enable_categorical=True)
    dva.set_float_info("label_lower_bound", lo_va)
    dva.set_float_info("label_upper_bound", hi_va)
    dte = xgb.DMatrix(Xte, enable_categorical=True)

    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": dist, "aft_loss_distribution_scale": scale,
        "tree_method": "hist", "learning_rate": 0.05, "max_depth": 4,
        "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 7,
    }
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                    early_stopping_rounds=40, verbose_eval=False)

    pred_t = bst.predict(dte)                 # predicted survival time (higher => lower risk)
    risk = -pred_t
    mu = np.log(np.clip(pred_t, 1e-6, None))  # AFT: log T = mu + scale*Z,  Z ~ dist
    z = (np.log(grid)[None, :] - mu[:, None]) / scale
    surv = 1.0 - norm.cdf(z)                  # S(t|x) = P(Z > z) for the normal AFT
    logger.info("XGB-AFT best_iteration=%s", getattr(bst, "best_iteration", "?"))
    return risk, surv, bst


def fit_xgb_cox(Xtr, Xva, Xte, t_tr, e_tr, t_va, e_va, rounds=600):
    """XGBoost Cox (risk score only) — concordance-oriented reference. Returns risk_test."""
    import xgboost as xgb

    def _signed(t, e):  # cox label: negative time => censored
        return np.where(e == 1, t, -t).astype(float)

    dtr = xgb.DMatrix(Xtr, label=_signed(t_tr, e_tr), enable_categorical=True)
    dva = xgb.DMatrix(Xva, label=_signed(t_va, e_va), enable_categorical=True)
    dte = xgb.DMatrix(Xte, enable_categorical=True)
    params = {
        "objective": "survival:cox", "eval_metric": "cox-nloglik",
        "tree_method": "hist", "learning_rate": 0.05, "max_depth": 4,
        "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 7,
    }
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                    early_stopping_rounds=40, verbose_eval=False)
    return bst.predict(dte)  # risk score, higher => higher risk


# --- orchestrate --------------------------------------------------------------
def run(features_path, meta_path, outdir):
    feats, meta, num, cat, (tr, e_tr, t_tr), (va, e_va, t_va), (te, e_te, t_te) = \
        load_split(features_path, meta_path)

    y_train = M.make_surv(e_tr, np.clip(t_tr, MIN_DURATION, None))
    y_test = M.make_surv(e_te, np.clip(t_te, MIN_DURATION, None))
    grid = M.time_grid(t_te, e_te, t_tr, n=20)
    logger.info("IBS time grid (min): %.1f .. %.1f over %d points", grid[0], grid[-1], len(grid))

    rows = []

    # Kaplan-Meier (IBS reference)
    try:
        km_sf = fit_km(np.clip(t_tr, MIN_DURATION, None), e_tr, grid)
        km_surv = np.tile(km_sf, (len(te), 1))
        rows.append(M.evaluate("Kaplan-Meier", y_train, y_test, e_te, t_te, t_tr,
                               surv_prob=km_surv, times=grid))
    except Exception as e:  # noqa: BLE001
        logger.warning("Kaplan-Meier failed: %s", e)

    # Cox PH
    try:
        cox_risk, cox_surv = fit_cox(tr, te, num, t_tr, e_tr, grid)
        rows.append(M.evaluate("Cox PH (lifelines)", y_train, y_test, e_te, t_te, t_tr,
                               risk=cox_risk, surv_prob=cox_surv, times=grid))
    except Exception as e:  # noqa: BLE001
        logger.warning("Cox PH failed: %s", e)

    # XGBoost AFT (the strong baseline)
    Xtr, Xva, Xte = _xy_xgb(tr, num, cat), _xy_xgb(va, num, cat), _xy_xgb(te, num, cat)
    aft_bst = None
    try:
        aft_risk, aft_surv, aft_bst = fit_xgb_aft(Xtr, Xva, Xte, t_tr, e_tr, t_va, e_va, grid)
        rows.append(M.evaluate("XGBoost AFT", y_train, y_test, e_te, t_te, t_tr,
                               risk=aft_risk, surv_prob=aft_surv, times=grid))
    except Exception as e:  # noqa: BLE001
        logger.warning("XGBoost AFT failed: %s", e)

    # XGBoost Cox (concordance reference)
    try:
        cox_xgb_risk = fit_xgb_cox(Xtr, Xva, Xte, t_tr, e_tr, t_va, e_va)
        rows.append(M.evaluate("XGBoost Cox", y_train, y_test, e_te, t_te, t_tr, risk=cox_xgb_risk))
    except Exception as e:  # noqa: BLE001
        logger.warning("XGBoost Cox failed: %s", e)

    _report(rows)
    _save(rows, meta, te, e_te, t_te, aft_bst, outdir)
    return rows


def _report(rows):
    logger.info("=" * 72)
    logger.info("SURVIVAL BASELINE RESULTS (test set)  --  higher C, lower IBS is better")
    logger.info("  %-22s %10s %10s %10s", "model", "C-Harrell", "C-IPCW", "IBS")
    for r in rows:
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) else "   n/a"
        logger.info("  %-22s %10s %10s %10s", r["model"], f(r["c_harrell"]), f(r["c_ipcw"]), f(r["ibs"]))
    logger.info("=" * 72)
    cs = [r["c_harrell"] for r in rows if isinstance(r["c_harrell"], float)]
    if cs:
        logger.info("BEST C-index to beat: %.4f", max(cs))


def _save(rows, meta, te, e_te, t_te, aft_bst, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "baseline_metrics.json").write_text(
        json.dumps({"results": rows, "n_test": int(len(te))}, indent=2, default=str), encoding="utf-8")
    logger.info("Saved metrics -> %s", outdir / "baseline_metrics.json")
    if aft_bst is not None:
        model_path = outdir / "xgb_aft.json"
        aft_bst.save_model(str(model_path))
        logger.info("Saved XGBoost AFT model -> %s", model_path)


def main():
    ap = argparse.ArgumentParser(description="Train + score classical survival baselines.")
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--meta", default="data/processed/feature_meta.json")
    ap.add_argument("--outdir", default="models/baselines")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _setup_logging(verbose=not args.quiet)
    run(args.features, args.meta, args.outdir)


if __name__ == "__main__":
    main()
