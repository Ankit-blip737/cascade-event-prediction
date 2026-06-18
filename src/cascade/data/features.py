"""
features.py — CASCADE Tier 0, Component #3: feature engineering.

Turns cleaned events (+ the junction graph) into a numeric feature matrix the XGBoost baseline,
the GCN+GRU trunk, and the DeepHit head all consume. Four feature families:

  A. Geo / identity   node_id (from graph.py snap), lat/lon, snap distance.
  B. Temporal         hour / day-of-week / month as cyclic sin-cos, weekend / night / peak flags,
                      and `is_holiday` (India-Karnataka public holidays — events spike on these).
  C. Categorical      event_type, event_cause, corridor, zone -> integer codes (+ saved mappings)
                      plus frequency encodings (a static "how common is this cause" prior).
  D. Ripple / contagion (CAUSAL, past-only)   the self-exciting signal the Hawkes head will model:
                      counts of prior incidents at the same node and at graph-neighbour nodes within
                      1h / 6h / 24h, time since the last incident at the node, and the node's running
                      historical road-closure rate. Every one of these uses ONLY events that started
                      strictly before the current event — no target leakage.

Targets passed through for downstream heads (survival convention from ingest.py):
  duration_min, event_observed (1=uncensored), censored, road_closure (0/1), priority_high (0/1/-1).

A chronological train/val/test `split` is added (70/15/15 by start time) so survival metrics and
conformal calibration respect time ordering (no peeking into the future).

Outputs (data/processed/):
  features.parquet       the model-ready matrix (one row per event)
  feature_meta.json      column roles (numeric / categorical / target), splits, category mappings

Usage:
    python -m src.cascade.data.features \
        --events data/processed/events_clean.parquet \
        --nodes  data/processed/events_nodes.parquet \
        --edges  data/processed/graph_edges.parquet \
        --outdir data/processed
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.features")

# --- columns ------------------------------------------------------------------
ID_COL = "id"
START_COL = "start_datetime"
LAT_COL, LON_COL = "latitude", "longitude"

# Ripple windows (minutes): the temporal scales over which incidents "excite" their neighbourhood.
RIPPLE_WINDOWS_MIN = [60, 360, 1440]   # 1h, 6h, 24h
NS_PER_MIN = 60_000_000_000            # 1 minute in nanoseconds
NO_PRIOR_SENTINEL_MIN = 7 * 24 * 60.0  # "time since last event" default when there is no prior

HOLIDAY_COUNTRY = "IN"
HOLIDAY_SUBDIV = "KA"                   # Karnataka


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _start_ns(df: pd.DataFrame) -> np.ndarray:
    """start_datetime as float nanoseconds (UTC); NaT -> NaN. The single time key used everywhere."""
    ts = pd.to_datetime(df[START_COL], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    t = ts.to_numpy().astype("datetime64[ns]").astype("int64").astype(float)
    t[ts.isna().to_numpy()] = np.nan
    return t


def _time_order(t_ns: np.ndarray) -> np.ndarray:
    """Stable chronological order of positions, NaN (NaT) pushed to the end — never returns -1."""
    key = np.where(np.isnan(t_ns), np.inf, t_ns)
    return np.argsort(key, kind="stable")


# --- load + merge -------------------------------------------------------------
def load_inputs(events_path: str | Path, nodes_path: str | Path) -> pd.DataFrame:
    """Cleaned events LEFT-JOIN the event->node mapping from graph.py (every event has a node)."""
    df = pd.read_parquet(events_path)
    en = pd.read_parquet(nodes_path)[[ID_COL, "node_id", "snap_dist_km", "snapped"]]
    df = df.merge(en, on=ID_COL, how="left", validate="one_to_one")
    missing = int(df["node_id"].isna().sum())
    if missing:
        logger.warning("%d events have no node_id after merge (graph.py out of sync?).", missing)
    logger.info("Loaded %d events, merged node assignments.", len(df))
    return df


# --- B. temporal --------------------------------------------------------------
def _cyc(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    ang = 2.0 * np.pi * values / period
    return np.sin(ang), np.cos(ang)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[START_COL], utc=True)
    # Bengaluru is UTC+5:30 — convert so "hour"/"is_night" reflect local clock, not UTC.
    local = ts.dt.tz_convert("Asia/Kolkata")

    hour = local.dt.hour.fillna(0).astype(int).to_numpy()
    dow = local.dt.dayofweek.fillna(0).astype(int).to_numpy()   # 0=Mon
    month = local.dt.month.fillna(1).astype(int).to_numpy()

    df["hour"] = hour
    df["dow"] = dow
    df["month"] = month
    df["hour_sin"], df["hour_cos"] = _cyc(hour, 24)
    df["dow_sin"], df["dow_cos"] = _cyc(dow, 7)
    df["month_sin"], df["month_cos"] = _cyc(month, 12)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_night"] = ((hour < 6) | (hour >= 22)).astype(int)
    df["is_morning_peak"] = ((hour >= 8) & (hour <= 11)).astype(int)
    df["is_evening_peak"] = ((hour >= 17) & (hour <= 20)).astype(int)

    # India / Karnataka public holidays (events spike around them)
    try:
        import holidays
        yrs = sorted(set(local.dt.year.dropna().astype(int).tolist()) or {2024})
        hol = holidays.country_holidays(HOLIDAY_COUNTRY, subdiv=HOLIDAY_SUBDIV, years=yrs)
        dates = local.dt.date
        df["is_holiday"] = dates.map(lambda d: int(d in hol) if pd.notna(d) else 0).astype(int)
        logger.info("Holiday flag set (%d holidays over %s).", len(hol), yrs)
    except Exception as e:  # noqa: BLE001 — never let an optional dep break feature build
        logger.warning("holidays unavailable (%s); is_holiday=0 for all rows.", type(e).__name__)
        df["is_holiday"] = 0
    return df


# --- C. categoricals ----------------------------------------------------------
def _encode(series: pd.Series, unknown: str = "unknown") -> tuple[np.ndarray, dict, np.ndarray]:
    """Label-encode a string column (null -> `unknown`). Returns (codes, name->code, freq_per_row)."""
    s = series.astype("string").fillna(unknown).str.strip().str.lower()
    cats = sorted(s.unique())
    mapping = {c: i for i, c in enumerate(cats)}
    codes = s.map(mapping).to_numpy()
    counts = s.value_counts()
    freq = s.map(counts).to_numpy().astype(float)
    return codes, mapping, freq


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    mappings: dict[str, dict] = {}

    df["event_type_planned"] = (df["event_type"].astype("string").str.strip().str.lower()
                                == "planned").astype(int)

    for col in ["event_cause", "corridor", "zone"]:
        codes, mapping, freq = _encode(df[col])
        df[f"{col}_code"] = codes
        df[f"{col}_freq"] = freq
        mappings[col] = mapping
    logger.info("Encoded categoricals: event_cause(%d), corridor(%d), zone(%d).",
                len(mappings["event_cause"]), len(mappings["corridor"]), len(mappings["zone"]))
    return df, mappings


# --- D. ripple / contagion (causal) ------------------------------------------
def _neighbors(edges_path: str | Path) -> dict[int, list[int]]:
    e = pd.read_parquet(edges_path)
    neigh: dict[int, set[int]] = {}
    for u, v in zip(e["u"].astype(int), e["v"].astype(int)):
        neigh.setdefault(u, set()).add(v)
        neigh.setdefault(v, set()).add(u)
    return {k: sorted(v) for k, v in neigh.items()}


def add_ripple_features(df: pd.DataFrame, edges_path: str | Path) -> pd.DataFrame:
    """
    Strictly-causal self-exciting features. For each event we look only at incidents that STARTED
    BEFORE it (right-open window), at the same node and at 1-hop graph neighbours.
    """
    df = df.copy()
    neigh = _neighbors(edges_path)

    # event start as float nanoseconds (UTC); NaT -> NaN so we can skip those rows
    t_ns = _start_ns(df)
    node = df["node_id"].to_numpy()

    # per-node SORTED arrays of all start times (we bound by time, so future rows never leak in)
    node_times: dict[int, np.ndarray] = {}
    valid = ~np.isnan(t_ns) & ~pd.isna(node)
    for n in np.unique(node[valid]):
        arr = np.sort(t_ns[valid & (node == n)])
        node_times[int(n)] = arr

    def _count_before(arr: np.ndarray, t: float, width_ns: float) -> int:
        """# events in arr within [t-width, t)  (strictly before t -> excludes self & ties)."""
        if arr.size == 0:
            return 0
        hi = np.searchsorted(arr, t, side="left")          # count of times < t
        lo = np.searchsorted(arr, t - width_ns, side="left")
        return int(hi - lo)

    n = len(df)
    out = {f"ripple_node_{w}": np.zeros(n) for w in RIPPLE_WINDOWS_MIN}
    out.update({f"ripple_neigh_{w}": np.zeros(n) for w in RIPPLE_WINDOWS_MIN})
    time_since = np.full(n, NO_PRIOR_SENTINEL_MIN, dtype=float)
    node_rank = np.zeros(n, dtype=float)  # how many prior events at this node (running count)

    widths_ns = {w: w * NS_PER_MIN for w in RIPPLE_WINDOWS_MIN}

    for i in range(n):
        t = t_ns[i]
        ni = node[i]
        if np.isnan(t) or pd.isna(ni):
            continue
        ni = int(ni)
        self_arr = node_times.get(ni, np.empty(0))
        for w in RIPPLE_WINDOWS_MIN:
            out[f"ripple_node_{w}"][i] = _count_before(self_arr, t, widths_ns[w])
            cnt = 0
            for m in neigh.get(ni, ()):
                cnt += _count_before(node_times.get(m, np.empty(0)), t, widths_ns[w])
            out[f"ripple_neigh_{w}"][i] = cnt

        prior = np.searchsorted(self_arr, t, side="left")  # # events strictly before t at this node
        node_rank[i] = prior
        if prior > 0:
            time_since[i] = min((t - self_arr[prior - 1]) / NS_PER_MIN, NO_PRIOR_SENTINEL_MIN)

    for k, v in out.items():
        df[k] = v
    df["time_since_last_node_min"] = time_since
    df["node_event_rank"] = node_rank
    df["has_prior_at_node"] = (node_rank > 0).astype(int)
    logger.info("Ripple features built (windows=%s min, causal/past-only).", RIPPLE_WINDOWS_MIN)
    return df


def add_node_history_closure_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Causal running road-closure rate at each node: mean of `requires_road_closure` over PRIOR
    events at the same node (shifted so the current row is excluded). Default = global prior.
    """
    df = df.copy()
    closure = df["requires_road_closure"].astype("boolean").fillna(False).astype(int).to_numpy()
    global_rate = float(closure.mean())

    node = pd.to_numeric(df["node_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    order = _time_order(_start_ns(df))

    rate = np.full(len(df), global_rate, dtype=float)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for pos in order:  # walk events in time order, keep per-node running closure stats (causal)
        nd = int(node[pos])
        c = counts.get(nd, 0)
        rate[pos] = (sums.get(nd, 0.0) / c) if c > 0 else global_rate  # strictly-prior mean
        counts[nd] = c + 1
        sums[nd] = sums.get(nd, 0.0) + closure[pos]
    df["node_closure_rate_prior"] = rate
    logger.info("Node prior closure-rate built (global prior=%.3f).", global_rate)
    return df


# --- F. targets ---------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["road_closure"] = df["requires_road_closure"].astype("boolean").fillna(False).astype(int)
    pr = df["priority"].astype("string").str.strip().str.lower()
    is_high = pr.eq("high").fillna(False).to_numpy()
    is_low = pr.eq("low").fillna(False).to_numpy()
    df["priority_high"] = np.select([is_high, is_low], [1, 0], default=-1)  # -1 = unknown
    df["censored"] = df["censored"].astype(int)
    df["log_duration_min"] = np.log1p(df["duration_min"].clip(lower=0))
    return df


# --- split --------------------------------------------------------------------
def add_temporal_split(df: pd.DataFrame, frac_train=0.70, frac_val=0.15) -> pd.DataFrame:
    """Chronological split: earliest events -> train, latest -> test. Invalid-duration rows -> 'drop'."""
    df = df.copy()
    order = _time_order(_start_ns(df))
    ranks = np.empty(len(df), dtype=float)
    ranks[order] = np.arange(len(df)) / len(df)
    split = np.where(ranks < frac_train, "train",
                     np.where(ranks < frac_train + frac_val, "val", "test"))
    if "valid_duration" in df.columns:
        split = np.where(df["valid_duration"].to_numpy(), split, "drop")
    df["split"] = split
    vc = pd.Series(split).value_counts().to_dict()
    logger.info("Temporal split: %s", vc)
    return df


# --- orchestrate --------------------------------------------------------------
NUMERIC_FEATURES = [
    "latitude", "longitude", "snap_dist_km",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_night", "is_morning_peak", "is_evening_peak", "is_holiday",
    "event_type_planned",
    "event_cause_freq", "corridor_freq", "zone_freq",
    *[f"ripple_node_{w}" for w in RIPPLE_WINDOWS_MIN],
    *[f"ripple_neigh_{w}" for w in RIPPLE_WINDOWS_MIN],
    "time_since_last_node_min", "node_event_rank", "has_prior_at_node",
    "node_closure_rate_prior",
]
CATEGORICAL_FEATURES = ["event_cause_code", "corridor_code", "zone_code"]
TARGETS = ["duration_min", "log_duration_min", "event_observed", "censored",
           "road_closure", "priority_high"]
KEEP_META = [ID_COL, "node_id", START_COL, "split", "snapped"]


def build_features(events_path, nodes_path, edges_path) -> tuple[pd.DataFrame, dict]:
    df = load_inputs(events_path, nodes_path)
    df = add_temporal_features(df)
    df, mappings = encode_categoricals(df)
    df = add_ripple_features(df, edges_path)
    df = add_node_history_closure_rate(df)
    df = add_targets(df)
    df = add_temporal_split(df)

    cols = KEEP_META + NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS
    cols = [c for c in cols if c in df.columns]
    feats = df[cols].copy()

    meta = {
        "numeric_features": [c for c in NUMERIC_FEATURES if c in feats.columns],
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in feats.columns],
        "targets": [c for c in TARGETS if c in feats.columns],
        "meta_cols": [c for c in KEEP_META if c in feats.columns],
        "category_mappings": mappings,
        "ripple_windows_min": RIPPLE_WINDOWS_MIN,
        "split_counts": feats["split"].value_counts().to_dict(),
        "n_rows": int(len(feats)),
    }
    return feats, meta


def save_features(feats: pd.DataFrame, meta: dict, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "features.parquet"
    try:
        feats.to_parquet(p, index=False)
        logger.info("Saved features -> %s (%d rows x %d cols)", p, len(feats), feats.shape[1])
    except Exception as e:  # noqa: BLE001
        csv = p.with_suffix(".csv")
        feats.to_csv(csv, index=False)
        logger.warning("Parquet failed (%s); saved CSV -> %s", e, csv)
    mp = outdir / "feature_meta.json"
    mp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved feature metadata -> %s", mp)


def _summary(feats: pd.DataFrame, meta: dict) -> None:
    logger.info("=" * 60)
    logger.info("FEATURE SUMMARY")
    logger.info("  rows .................... %d", len(feats))
    logger.info("  numeric features ........ %d", len(meta["numeric_features"]))
    logger.info("  categorical features .... %d", len(meta["categorical_features"]))
    logger.info("  targets ................. %s", meta["targets"])
    logger.info("  split counts ............ %s", meta["split_counts"])
    nn = int(feats[meta["numeric_features"]].isna().any(axis=1).sum())
    logger.info("  rows with any NaN feat .. %d", nn)
    logger.info("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the CASCADE feature matrix.")
    ap.add_argument("--events", default="data/processed/events_clean.parquet")
    ap.add_argument("--nodes", default="data/processed/events_nodes.parquet")
    ap.add_argument("--edges", default="data/processed/graph_edges.parquet")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _setup_logging(verbose=not args.quiet)
    for pth in (args.events, args.nodes, args.edges):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Missing input: {pth} (run ingest.py then graph.py first)")

    feats, meta = build_features(args.events, args.nodes, args.edges)
    save_features(feats, meta, args.outdir)
    _summary(feats, meta)


if __name__ == "__main__":
    main()
