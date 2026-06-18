"""
aci.py — CASCADE Tier 0, Component #11b: Adaptive Conformal Inference (shift-safe coverage).

Split conformal (conformal_survival.py) guarantees coverage only under exchangeability — which the
closed loop and temporal drift break (CLAUDE.md patch #1). Adaptive Conformal Inference
(Gibbs & Candes, 2021) restores it online: after each incident it nudges the working miscoverage
level toward the target,

        alpha_{t+1} = clip( alpha_t + gamma * (alpha_target - err_t) ),

where err_t = 1 if the realized duration fell OUTSIDE the bound, else 0. When the stream drifts and
coverage slips, alpha shrinks (wider, safer bounds) automatically; when it over-covers, alpha grows
(tighter bounds). Long-run coverage converges to 1 - alpha_target regardless of the shift.

Here we stream the TEST incidents in time order, forming an upper bound from the calibration
nonconformity scores at the current adaptive level, and show the running coverage tracking the
target — versus a fixed (non-adaptive) bound that can drift off.

Usage:
    python -m src.cascade.calibrate.aci --preds models/preds_all.npz --alpha 0.1 --gamma 0.02
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.cascade.calibrate.conformal_survival import predict_quantile

logger = logging.getLogger("cascade.aci")


def _setup_logging(verbose=True):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


class ACI:
    """Online miscoverage controller. `offset(a)` maps a working alpha to a calibration-score offset."""

    def __init__(self, alpha_target=0.1, gamma=0.02):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha = alpha_target

    def offset(self, cal_scores, a):
        a = float(np.clip(a, 1e-3, 1 - 1e-3))
        return float(np.quantile(cal_scores, 1 - a, method="higher"))

    def update(self, covered: bool):
        err = 0.0 if covered else 1.0
        self.alpha = float(np.clip(self.alpha + self.gamma * (self.alpha_target - err), 1e-3, 1 - 1e-3))
        return self.alpha


def run(preds_path, alpha, gamma):
    p = np.load(preds_path, allow_pickle=True)
    pmf, cuts, split = p["pmf"], p["cuts"], p["split"]
    dur, ev = p["durations"], p["events"]

    cal = split == 1
    te = split == 2
    # fixed target-level predicted quantile + calibration nonconformity scores (uncensored)
    qhat_cal = predict_quantile(pmf[cal], cuts, 1 - alpha)
    unc_cal = ev[cal].astype(bool)
    cal_scores = dur[cal][unc_cal] - qhat_cal[unc_cal]

    # test stream in time order (preds are already time-sorted), uncensored only (need realized T)
    te_idx = np.where(te)[0]
    unc_te = ev[te_idx].astype(bool)
    te_idx = te_idx[unc_te]
    qhat_te = predict_quantile(pmf[te_idx], cuts, 1 - alpha)
    T_te = dur[te_idx]

    aci = ACI(alpha_target=alpha, gamma=gamma)
    fixed_off = aci.offset(cal_scores, alpha)        # non-adaptive baseline offset

    cov_aci = np.empty(len(te_idx), dtype=bool)
    cov_fix = np.empty(len(te_idx), dtype=bool)
    alphas = np.empty(len(te_idx))
    for i in range(len(te_idx)):
        alphas[i] = aci.alpha
        bound_aci = qhat_te[i] + aci.offset(cal_scores, aci.alpha)
        bound_fix = qhat_te[i] + fixed_off
        cov_aci[i] = T_te[i] <= bound_aci
        cov_fix[i] = T_te[i] <= bound_fix
        aci.update(cov_aci[i])

    logger.info("=" * 64)
    logger.info("ADAPTIVE CONFORMAL INFERENCE  (target %.0f%% coverage, gamma=%.3f)",
                100 * (1 - alpha), gamma)
    logger.info("  test uncensored incidents streamed ... %d", len(te_idx))
    logger.info("  ACI   running coverage ............... %.1f%%", 100 * cov_aci.mean())
    logger.info("  fixed running coverage ............... %.1f%%", 100 * cov_fix.mean())
    logger.info("  alpha drifted %.3f -> %.3f (auto-adjusts to drift)", alpha, aci.alpha)
    logger.info("=" * 64)
    return cov_aci, cov_fix, alphas


def main():
    ap = argparse.ArgumentParser(description="Adaptive Conformal Inference over the test stream.")
    ap.add_argument("--preds", default="models/preds_all.npz")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    if not Path(args.preds).exists():
        raise FileNotFoundError(f"Missing {args.preds}")
    run(args.preds, args.alpha, args.gamma)


if __name__ == "__main__":
    main()
