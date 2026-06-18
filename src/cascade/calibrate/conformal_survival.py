"""
conformal_survival.py — CASCADE Tier 0, Component #11a: calibrated duration bounds.

The DeepHit head outputs a duration PMF, but "P90 = 95 min" is only trustworthy if 90% really means
90%. This module wraps the PMF in split-conformal calibration so the bounds carry an *empirical
coverage guarantee* — the property the XGBoost baseline has no notion of, and CASCADE's headline
safety feature.

Two bounds per incident (CQR-style, conformalized):
  * UPPER  U(x): "with prob >= 1-alpha the incident clears within U(x) minutes" — the actionable
                 worst-case clearance time. Calibrated on UNCENSORED calibration points (an upper
                 bound needs the true end time).
  * LOWER  L(x): "with prob >= 1-alpha it lasts at least L(x) minutes". Right-censoring is FAVOURABLE
                 for a lower bound (a censored Y=min(T,C) <= T only makes the score conservative), so
                 all calibration points are usable.

Calibration set = the validation split (disjoint from train; per CLAUDE.md it must be re-drawn after
every retrain). Coverage is then checked on the test split. See aci.py for the shift-safe online
version (Adaptive Conformal Inference).

Severity for the allocator = the calibrated upper bound (worst-case clearance minutes): a long
guaranteed clearance time is what makes a junction worth barricading / staffing.

Usage:
    python -m src.cascade.calibrate.conformal_survival \
        --preds models/preds_all.npz --alpha 0.1 --out models/calibrated.npz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("cascade.conformal")


def _setup_logging(verbose=True):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


def predict_quantile(pmf: np.ndarray, cuts: np.ndarray, tau: float) -> np.ndarray:
    """Duration at which the predicted CIF first reaches `tau`, linearly interpolated within the bin."""
    cum = np.cumsum(pmf, axis=1)          # CIF at each bin's right edge
    left, right = cuts[:-1], cuts[1:]
    out = np.empty(len(pmf))
    for i in range(len(pmf)):
        k = int(np.searchsorted(cum[i], tau))
        if k >= len(right):
            out[i] = right[-1]
        else:
            c_lo = cum[i, k - 1] if k > 0 else 0.0
            frac = (tau - c_lo) / max(cum[i, k] - c_lo, 1e-9)
            out[i] = left[k] + frac * (right[k] - left[k])
    return out


class ConformalDuration:
    """Split-conformal lower/upper bounds on incident duration, calibrated on a held-out set."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, cal_pmf, cuts, cal_obs_time, cal_event):
        self.cuts = cuts
        unc = cal_event.astype(bool)
        n_up = int(unc.sum())
        lvl_up = min(1.0, (1 - self.alpha) * (1 + 1.0 / max(n_up, 1)))   # finite-sample correction

        q_hi = predict_quantile(cal_pmf, cuts, 1 - self.alpha)
        s_up = cal_obs_time[unc] - q_hi[unc]                              # CQR upper nonconformity
        self.q_up = float(np.quantile(s_up, lvl_up, method="higher"))

        n_lo = len(cal_obs_time)
        lvl_lo = min(1.0, (1 - self.alpha) * (1 + 1.0 / n_lo))
        q_lo = predict_quantile(cal_pmf, cuts, self.alpha)
        s_lo = q_lo - cal_obs_time                                       # censoring-favourable
        self.q_lo = float(np.quantile(s_lo, lvl_lo, method="higher"))
        logger.info("Calibrated offsets: upper +%.1f min (n_unc=%d), lower -%.1f min (n=%d)",
                    self.q_up, n_up, self.q_lo, n_lo)
        return self

    def upper(self, pmf):
        return predict_quantile(pmf, self.cuts, 1 - self.alpha) + self.q_up

    def lower(self, pmf):
        return np.clip(predict_quantile(pmf, self.cuts, self.alpha) - self.q_lo, 0, None)

    def median(self, pmf):
        return predict_quantile(pmf, self.cuts, 0.5)


def coverage_upper(upper, obs_time, event):
    """Empirical P(T <= U) on uncensored events. Should be >= 1-alpha if calibration worked."""
    unc = event.astype(bool)
    return float((obs_time[unc] <= upper[unc]).mean())


def run(preds_path, alpha, out_path):
    p = np.load(preds_path, allow_pickle=True)
    pmf, cuts, split = p["pmf"], p["cuts"], p["split"]
    dur, ev = p["durations"], p["events"]

    cal = split == 1   # validation = calibration (disjoint from train)
    te = split == 2
    cd = ConformalDuration(alpha=alpha).fit(pmf[cal], cuts, dur[cal], ev[cal])

    lower_all, upper_all, med_all = cd.lower(pmf), cd.upper(pmf), cd.median(pmf)

    cov_te = coverage_upper(upper_all[te], dur[te], ev[te])
    cov_cal = coverage_upper(upper_all[cal], dur[cal], ev[cal])
    logger.info("=" * 64)
    logger.info("CONFORMAL CALIBRATION  (target coverage %.0f%%)", 100 * (1 - alpha))
    logger.info("  upper-bound coverage  calib=%.1f%%   test=%.1f%%", 100 * cov_cal, 100 * cov_te)
    logger.info("  test median predicted clearance: %.0f min", float(np.median(med_all[te])))
    logger.info("  test median P%.0f upper bound:    %.0f min",
                100 * (1 - alpha), float(np.median(upper_all[te])))
    logger.info("=" * 64)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if "node_intensity" in p.files:      # Tier-1: predicted per-junction next-incident rate
        extra["node_intensity"] = p["node_intensity"]
    if "intensity" in p.files:
        extra["intensity"] = p["intensity"]
    np.savez_compressed(out_path,
                        event_id=p["event_id"], node_id=p["node_id"], split=split,
                        lower=lower_all, upper=upper_all, median=med_all,
                        closure_prob=p["closure_prob"], priority_prob=p["priority_prob"],
                        durations=dur, events=ev, alpha=alpha, **extra)
    logger.info("Saved calibrated bounds -> %s", out_path)
    return cd


def main():
    ap = argparse.ArgumentParser(description="Conformal calibration of DeepHit duration bounds.")
    ap.add_argument("--preds", default="models/preds_all.npz")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage (0.1 -> 90% bounds)")
    ap.add_argument("--out", default="models/calibrated.npz")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    if not Path(args.preds).exists():
        raise FileNotFoundError(f"Missing {args.preds} (download preds_all.npz from Colab first)")
    run(args.preds, args.alpha, args.out)


if __name__ == "__main__":
    main()
