"""
metrics.py — CASCADE Tier 0: survival evaluation metrics (the "number to beat" yardstick).

Two metrics, both censoring-aware (scikit-survival):
  * Concordance index (C-index)   — ranking quality: does the model order incidents by risk
                                    correctly? Harrell's + the IPCW-corrected variant (robust under
                                    censoring). Higher is better; 0.5 == random.
  * Integrated Brier Score (IBS)  — calibration of the predicted survival curves over a time grid.
                                    Lower is better; the Kaplan-Meier marginal is the naive reference.

Survival convention (matches ingest.py): event_observed=1 => uncensored (the incident ended);
0 => censored (still active at last observation). Durations are in MINUTES.

These functions are model-agnostic — they take risk scores / survival-probability matrices, so the
exact same yardstick scores the XGBoost baseline today and the DeepHit head later.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("cascade.metrics")


def make_surv(event_observed: np.ndarray, duration_min: np.ndarray):
    """Build a scikit-survival structured array y = (event:bool, time:float)."""
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(event_observed).astype(bool),
                            time=np.asarray(duration_min, dtype=float))


def time_grid(test_time: np.ndarray, test_event: np.ndarray,
              train_time: np.ndarray, n: int = 20) -> np.ndarray:
    """
    Evaluation time grid for IBS: spread between the 5th and 95th percentile of UNCENSORED test
    durations, then clipped strictly inside the follow-up of both train and test (sksurv requires
    the censoring distribution to be estimable at every grid point).
    """
    ev = np.asarray(test_time)[np.asarray(test_event).astype(bool)]
    if ev.size == 0:
        ev = np.asarray(test_time)
    lo, hi = np.percentile(ev, 5), np.percentile(ev, 95)
    upper = min(float(np.max(test_time)), float(np.max(train_time)))
    lower = max(float(np.min(test_time)), float(np.min(train_time)))
    lo = max(lo, lower + 1e-3)
    hi = min(hi, upper - 1e-3)
    if not (hi > lo):  # degenerate fallback
        lo, hi = lower + 1e-3, upper - 1e-3
    return np.linspace(lo, hi, n)


def c_index_harrell(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance. `risk` higher => higher hazard => shorter survival."""
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(
        np.asarray(event).astype(bool), np.asarray(time, float), np.asarray(risk, float))[0])


def c_index_ipcw(train_surv, test_surv, risk: np.ndarray, tau: float | None = None) -> float:
    """IPCW-corrected concordance (Uno) — less biased than Harrell's under heavy censoring."""
    from sksurv.metrics import concordance_index_ipcw

    return float(concordance_index_ipcw(train_surv, test_surv, np.asarray(risk, float), tau=tau)[0])


def integrated_brier(train_surv, test_surv, surv_prob: np.ndarray, times: np.ndarray) -> float:
    """
    Integrated Brier Score. `surv_prob` is [n_test, n_times] = predicted P(T > t) at each grid time.
    Lower is better.
    """
    from sksurv.metrics import integrated_brier_score

    return float(integrated_brier_score(train_surv, test_surv, surv_prob, times))


def evaluate(name: str, train_surv, test_surv, test_event, test_time, train_time,
             risk: np.ndarray | None = None, surv_prob: np.ndarray | None = None,
             times: np.ndarray | None = None) -> dict:
    """Score one model. Any metric whose inputs are missing is recorded as None (never crashes)."""
    res: dict[str, float | None] = {"model": name, "c_harrell": None, "c_ipcw": None, "ibs": None}
    if risk is not None:
        try:
            res["c_harrell"] = c_index_harrell(test_event, test_time, risk)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] Harrell C failed: %s", name, e)
        try:
            res["c_ipcw"] = c_index_ipcw(train_surv, test_surv, risk,
                                         tau=float(times.max()) if times is not None else None)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] IPCW C failed: %s", name, e)
    if surv_prob is not None and times is not None:
        try:
            res["ibs"] = integrated_brier(train_surv, test_surv, surv_prob, times)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] IBS failed: %s", name, e)
    return res
