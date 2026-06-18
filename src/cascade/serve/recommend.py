"""
recommend.py — CASCADE: the serving orchestrator (one operational recommendation).

Fuses every computed layer — calibrated predictions, the OR-Tools allocation (officers/barricades),
diversions, the digital-twin impact + economics, the GATE-PPO dispatcher, and the closed-loop
signals — into a single `recommendation.json` payload for the dashboard/API, plus a plain-text ops
briefing grounded ONLY in the computed numbers (the slot a Gemini briefing can later render verbatim).

Reads the saved artifacts under models/; does not re-run solvers (each stage owns its module), so it
stays light and free of the pyarrow/OR-Tools/torch load-order traps.

Usage:
    python -m src.cascade.serve.recommend
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("cascade.serve")


def _setup_logging(v=True):
    logging.basicConfig(level=logging.INFO if v else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")


def _load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build(models_dir="models"):
    m = Path(models_dir)
    alloc = _load(m / "allocation.json")
    div = _load(m / "diversions.json")
    twin = _load(m / "twin_report.json")
    disp = _load(m / "dispatcher_report.json")
    loop = _load(m / "closed_loop_report.json")
    ev = _load(m / "final_eval.json")

    officers = (alloc or {}).get("officers", [])
    barricades = (alloc or {}).get("barricades", [])
    rec = {
        "summary": {
            "officer_teams": len(officers),
            "barricades": len(barricades),
            "diversions": len(div or []),
            "officer_coverage_pct": round(100 * (alloc or {}).get("coverage", {}).get("covered_fraction", 0), 1),
            "twin_congestion_saved_pct": round(100 * (twin or {}).get("totals", {}).get("saved_veh_min", 0)
                                               / max((twin or {}).get("totals", {}).get("baseline_veh_min", 1), 1), 1) if twin else None,
            "twin_inr_per_day": (twin or {}).get("totals", {}).get("inr_per_day"),
            "dispatcher_lift_vs_greedy_pct": (disp or {}).get("lift_vs_greedy_pct"),
            "offpolicy_lift_vs_random_pct": (loop or {}).get("off_policy_reward", {}).get("lift_vs_random_pct"),
        },
        "deploy_officers": officers,
        "barricades": barricades,
        "diversions": div or [],
        "impact": (twin or {}).get("totals"),
        "model_eval": ev,
        "closed_loop": loop,
    }
    return rec


def briefing(rec):
    """Plain-text daily ops briefing, grounded strictly in the computed recommendation."""
    s = rec["summary"]
    lines = ["CASCADE - DAILY OPS BRIEFING", "=" * 40]
    lines.append(f"Deploy {s['officer_teams']} officer teams covering {s['officer_coverage_pct']}% of expected "
                 f"congestion-minutes within response range.")
    if rec["deploy_officers"][:3]:
        tops = ", ".join(f"{o['junction']} ({o['headcount']})" for o in rec["deploy_officers"][:3])
        lines.append(f"  Priority junctions: {tops}.")
    lines.append(f"Barricade {s['barricades']} closure-prone junctions; {s['diversions']} diversions prepared.")
    if rec["barricades"][:3]:
        b = ", ".join(x["junction"] for x in rec["barricades"][:3])
        lines.append(f"  Top barricades: {b}.")
    if s.get("twin_congestion_saved_pct") is not None:
        lines.append(f"Projected impact: ~{s['twin_congestion_saved_pct']}% congestion avoided "
                     f"(~Rs {int(s['twin_inr_per_day']):,}/day) [illustrative twin].")
    if s.get("dispatcher_lift_vs_greedy_pct") is not None:
        lines.append(f"RL dispatcher relieves {s['dispatcher_lift_vs_greedy_pct']}% more congestion than greedy.")
    if s.get("offpolicy_lift_vs_random_pct") is not None:
        lines.append(f"On realized outcomes the plan beats random placement by "
                     f"{s['offpolicy_lift_vs_random_pct']}%.")
    return "\n".join(lines)


def run(models_dir, out_path):
    rec = build(models_dir)
    rec["ops_briefing"] = briefing(rec)
    Path(out_path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    logger.info("=" * 64)
    for ln in rec["ops_briefing"].splitlines():
        logger.info("  %s", ln)
    logger.info("=" * 64)
    logger.info("Saved unified recommendation -> %s", out_path)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Fuse all layers into one operational recommendation.")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="models/recommendation.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    run(args.models, args.out)


if __name__ == "__main__":
    main()
