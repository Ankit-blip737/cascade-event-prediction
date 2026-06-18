"""
ingest.py — CASCADE Tier 0, Component #1: raw ASTRAM event-log ingestion & cleaning.

Reads the raw ASTRAM incident CSV and produces a clean, SURVIVAL-READY DataFrame:
  - parses every timestamp column (tz-aware, errors coerced),
  - computes `duration_min` = observed clearance time, OR censoring time for active events,
  - sets `censored` / `event_observed` flags (survival convention: event_observed=1 => uncensored),
  - flags messy rows (missing start, negative or outlier durations) instead of silently dropping them.

Output: cleaned parquet at data/processed/events_clean.parquet + a printed summary.

Usage:
    python -m src.cascade.data.ingest \
        --input  data/raw/astram_events.csv \
        --output data/processed/events_clean.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.ingest")

# --- ASTRAM schema constants --------------------------------------------------
ID_COL = "id"
STATUS_COL = "status"
START_COL = "start_datetime"

# All timestamp columns we attempt to parse.
DATETIME_COLS = [
    "start_datetime", "end_datetime", "modified_datetime",
    "created_date", "closed_datetime", "resolved_datetime",
]

# End-time is coalesced in this priority order. `modified_datetime` is the last-resort
# proxy because many "closed" rows have NO closed_datetime but always have a modified time.
END_CANDIDATES = ["resolved_datetime", "closed_datetime", "end_datetime", "modified_datetime"]

# For active (censored) events, this is the "as of" time we last observed them.
CENSOR_TIME_COL = "modified_datetime"

ACTIVE_STATUS = "active"  # status value that means the incident has NOT ended -> censored

# Anything longer than this is flagged as a likely data error (kept, not dropped).
MAX_REASONABLE_DURATION_MIN = 60 * 24 * 30  # 30 days


# --- helpers ------------------------------------------------------------------
def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip column names so minor header inconsistencies don't break us."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV robustly: try utf-8, fall back to latin-1, never hard-fail on a bad line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    last_err: Exception | None = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc, on_bad_lines="warn")
            logger.info("Loaded %d rows x %d cols from '%s' (encoding=%s)",
                        len(df), df.shape[1], path.name, enc)
            return _normalize_columns(df)
        except UnicodeDecodeError as e:  # try the next encoding
            last_err = e
            logger.warning("Encoding '%s' failed, retrying...", enc)
    raise RuntimeError(f"Could not decode {path} with utf-8 or latin-1") from last_err


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse every known timestamp column to tz-aware UTC datetimes; bad values -> NaT."""
    df = df.copy()
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            n_bad = df[col].isna().sum()
            if n_bad:
                logger.info("  %-18s parsed (%d unparseable/empty -> NaT)", col, n_bad)
        else:
            logger.warning("  expected datetime column '%s' is missing", col)
    return df


def compute_duration_and_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core survival logic. Adds:
      status_norm       : lowercased status
      censored          : True if the incident has not ended (status == 'active', or no end recorded)
      event_observed    : 1 if uncensored (event happened), 0 if censored   <- survival convention
      end_dt_effective  : observed end time, or censoring time for active events
      duration_min      : (end_dt_effective - start) in minutes
      valid_duration    : True if start present and duration >= 0
      outlier_duration  : True if duration exceeds the sanity cap
    """
    df = df.copy()
    start = df[START_COL]

    # 1) normalize status
    status = (df[STATUS_COL].astype(str).str.strip().str.lower()
              if STATUS_COL in df.columns else pd.Series("", index=df.index))
    df["status_norm"] = status

    # 2) coalesce a single end time from the candidate columns (first non-null wins)
    end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in END_CANDIDATES:
        if col in df.columns:
            end = end.fillna(df[col])

    # 3) censoring: 'active' status is censored; also treat unknown-status-with-no-end as censored
    censored = status.eq(ACTIVE_STATUS) | (status.eq("") & end.isna())
    df["censored"] = censored
    df["event_observed"] = (~censored).astype(int)  # 1 = event happened (uncensored)

    # 4) effective end time:
    #    - censored rows  -> censoring time (last time we saw the still-active event)
    #    - uncensored     -> the coalesced observed end time
    censor_time = (df[CENSOR_TIME_COL] if CENSOR_TIME_COL in df.columns
                   else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]"))
    eff_end = end.where(~censored, other=censor_time)
    # safety net: an "ended" row that still has no end timestamp falls back to the modified time
    eff_end = eff_end.fillna(censor_time)
    df["end_dt_effective"] = eff_end

    # 5) duration in minutes
    df["duration_min"] = (eff_end - start).dt.total_seconds() / 60.0

    # 6) validity / outlier flags (we flag, we do NOT silently drop)
    df["valid_duration"] = start.notna() & df["duration_min"].notna() & (df["duration_min"] >= 0)
    df["outlier_duration"] = df["duration_min"] > MAX_REASONABLE_DURATION_MIN
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by id (keep the most recently modified record) and report what we found."""
    df = df.copy()
    before = len(df)

    if ID_COL in df.columns:
        sort_col = CENSOR_TIME_COL if CENSOR_TIME_COL in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col)
        df = df.drop_duplicates(subset=[ID_COL], keep="last")
        if len(df) != before:
            logger.info("Dropped %d duplicate id rows (kept latest by %s)", before - len(df), sort_col)

    return df.reset_index(drop=True)


def load_and_clean(path: str | Path, drop_invalid: bool = False) -> pd.DataFrame:
    """Public API: raw CSV path -> clean, survival-ready DataFrame."""
    df = load_raw(path)
    df = parse_datetimes(df)
    df = compute_duration_and_censoring(df)
    df = clean(df)

    if drop_invalid:
        n = len(df)
        df = df[df["valid_duration"]].reset_index(drop=True)
        logger.info("Dropped %d rows with invalid duration (drop_invalid=True)", n - len(df))

    _summary(df)
    return df


def _summary(df: pd.DataFrame) -> None:
    """Print a quick sanity report — read this every run to catch data issues early."""
    n = len(df)
    cens = int(df["censored"].sum())
    inval = int((~df["valid_duration"]).sum())
    outl = int(df["outlier_duration"].sum())
    obs = df.loc[df["valid_duration"] & (df["event_observed"] == 1), "duration_min"]
    logger.info("=" * 60)
    logger.info("INGEST SUMMARY")
    logger.info("  rows total .............. %d", n)
    logger.info("  censored (active) ....... %d (%.1f%%)", cens, 100 * cens / max(n, 1))
    logger.info("  invalid duration ........ %d", inval)
    logger.info("  outlier duration (>30d) . %d", outl)
    if len(obs):
        logger.info("  observed duration (min): median=%.1f  p90=%.1f  max=%.1f",
                    obs.median(), obs.quantile(0.90), obs.max())
    logger.info("=" * 60)


def _save(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save to parquet (preserves dtypes); fall back to CSV if pyarrow is unavailable."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
        logger.info("Saved cleaned data -> %s", out_path)
    except Exception as e:  # pyarrow missing or parquet error -> csv fallback
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning("Parquet failed (%s); saved CSV -> %s", e, csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest & clean the raw ASTRAM event CSV.")
    ap.add_argument("--input", required=True, help="path to raw ASTRAM CSV")
    ap.add_argument("--output", default="data/processed/events_clean.parquet",
                    help="output parquet path")
    ap.add_argument("--drop-invalid", action="store_true",
                    help="drop rows with missing start / negative duration")
    ap.add_argument("--quiet", action="store_true", help="reduce logging")
    args = ap.parse_args()

    _setup_logging(verbose=not args.quiet)
    df = load_and_clean(args.input, drop_invalid=args.drop_invalid)
    _save(df, args.output)


if __name__ == "__main__":
    main()
