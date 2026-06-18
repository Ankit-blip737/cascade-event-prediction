"""
dataset.py — CASCADE Tier 0, Component #7: assemble the Colab training bundle (torch-free).

This is the LOCAL half of the Colab bridge. It packs the cleaned features + junction graph into a
single self-describing `.npz` (+ JSON meta) that the GPU notebook loads with one `np.load`. No torch
required here — only numpy/pandas — so it runs on the weak local machine in seconds.

What goes in the bundle (rows are ALL non-dropped events, sorted ascending by start time, so the
temporal train/val/test split is just contiguous blocks):

  X_num        [N, Fn]  float32   standardized numeric features (z-scored on TRAIN stats only)
  X_cat        [N, Fc]  int64     categorical codes (event_cause / corridor / zone) for embeddings
  node_id      [N]      int64     which junction node each event sits on (0..num_nodes-1)
  hist_idx     [N, L]   int64     CAUSAL history: row indices of the up-to-L most recent prior events
                                  at the SAME node, left-padded with -1 (the GRU's input sequence)
  node_feat    [V, Fv]  float32   static per-junction features for the GCN (train-only aggregates)
  edge_index   [2, 2E]  int64     directed adjacency (both directions) from graph.npz
  durations    [N]      float32   duration_min (survival time)
  events       [N]      int64     event_observed (1=uncensored, 0=censored)
  bin_idx      [N]      int64     DeepHit time-bin index (cuts from TRAIN uncensored durations)
  cuts         [K+1]    float32   bin edges in minutes
  road_closure [N]      int64     aux head target: requires_road_closure
  priority_high[N]      int64     aux head target: priority (1 High / 0 Low / -1 unknown)
  split        [N]      int8      0=train 1=val 2=test
  event_id     [N]      <U..      original event id (to map predictions back for serve/recommend)

Handoff: upload `train_bundle.npz` (and nothing else — graph is embedded) to Colab; the notebook
writes weights back to `models/trunk_deephit.pt`.

Usage:
    python -m src.cascade.data.dataset \
        --features data/processed/features.parquet \
        --meta     data/processed/feature_meta.json \
        --graph    data/processed/graph.npz \
        --edges    data/processed/graph_edges.parquet \
        --out      data/processed/train_bundle.npz
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.dataset")

HIST_LEN = 8        # L: recent events per node fed to the GRU
N_BINS = 20         # K: DeepHit discrete time bins
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --- DeepHit discretization ---------------------------------------------------
def make_cuts(train_durations: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges from quantiles of TRAIN uncensored durations (equidistant in probability)."""
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(train_durations, qs)
    cuts = np.unique(cuts)                      # collapse ties in the heavy tail
    cuts[0] = min(cuts[0], float(train_durations.min()))
    cuts[-1] = max(cuts[-1], float(train_durations.max()))
    return cuts.astype(np.float32)


def to_bins(durations: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Map a duration to its bin 0..K-1 (right-closed; clipped into range)."""
    idx = np.searchsorted(cuts, durations, side="right") - 1
    return np.clip(idx, 0, len(cuts) - 2).astype(np.int64)


# --- causal history sequences -------------------------------------------------
def build_history(node: np.ndarray, n: int, length: int) -> np.ndarray:
    """
    For each row i (already time-sorted), the up-to-`length` most recent EARLIER rows at the same
    node, left-padded with -1 (chronological order: oldest..newest). Pure past -> no leakage.
    """
    hist = np.full((n, length), -1, dtype=np.int64)
    recent: dict[int, deque] = {}
    for i in range(n):
        nd = int(node[i])
        dq = recent.get(nd)
        if dq:
            seq = list(dq)[-length:]
            hist[i, length - len(seq):] = seq
        if dq is None:
            dq = deque(maxlen=length)
            recent[nd] = dq
        dq.append(i)
    return hist


# --- point-process (event-genesis) targets ------------------------------------
_NS_PER_MIN = 60_000_000_000


def build_pointprocess_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Tier-1 self-exciting head target: time to the NEXT incident at the SAME junction.

    For each event (rows already time-sorted) we look forward to the next event at its node:
      tte_min      : minutes until that next incident (the inter-arrival the Hawkes head models)
      tte_observed : 1 if a next incident exists at the node, else 0 (right-censored at end-of-data)
    Censored rows get the gap from their start to the last observed time in the dataset.
    """
    ts = pd.to_datetime(df["start_datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    t_ns = ts.to_numpy().astype("datetime64[ns]").astype("int64").astype(float)
    g = pd.DataFrame({"node": df["node_id"].to_numpy(), "t": t_ns})
    next_t = g.groupby("node")["t"].shift(-1).to_numpy()      # next event time at same node
    observed = ~np.isnan(next_t)
    end_of_data = np.nanmax(t_ns)
    gap = np.where(observed, next_t - t_ns, end_of_data - t_ns)
    tte_min = np.clip(gap / _NS_PER_MIN, 0.1, None).astype(np.float32)
    return tte_min, observed.astype(np.int64)


# --- static node features for the GCN -----------------------------------------
def build_node_features(train_df: pd.DataFrame, num_nodes: int, edges_path) -> tuple[np.ndarray, list]:
    """Per-junction static features from TRAIN events only (+ graph degree). Leak-safe."""
    tdf = train_df.copy()
    tdf["_prio"] = tdf["priority_high"].clip(lower=0)  # treat unknown(-1) as 0 for the node prior
    g = tdf.groupby("node_id")
    agg = pd.DataFrame({
        "n_events": g.size(),
        "closure_rate": g["road_closure"].mean(),
        "priority_rate": g["_prio"].mean(),
        "planned_rate": g["event_type_planned"].mean(),
        "lat": g["latitude"].median(),
        "lon": g["longitude"].median(),
    })
    full = pd.DataFrame(index=np.arange(num_nodes))
    full = full.join(agg)
    # global priors for nodes unseen in train
    full["n_events"] = full["n_events"].fillna(0.0)
    full["closure_rate"] = full["closure_rate"].fillna(train_df["road_closure"].mean())
    full["priority_rate"] = full["priority_rate"].fillna(train_df["priority_high"].clip(lower=0).mean())
    full["planned_rate"] = full["planned_rate"].fillna(train_df["event_type_planned"].mean())
    full["lat"] = full["lat"].fillna(train_df["latitude"].median())
    full["lon"] = full["lon"].fillna(train_df["longitude"].median())

    # graph degree
    e = pd.read_parquet(edges_path)
    deg = pd.concat([e["u"], e["v"]]).value_counts()
    full["degree"] = full.index.map(deg).fillna(0).astype(float)

    full["log_n_events"] = np.log1p(full["n_events"])
    cols = ["log_n_events", "closure_rate", "priority_rate", "planned_rate", "lat", "lon", "degree"]
    M = full[cols].to_numpy(dtype=np.float64)
    # standardize columns (robust to scale; lat/lon/degree differ wildly)
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    M = (M - mu) / sd
    return M.astype(np.float32), cols


# --- main assembly ------------------------------------------------------------
def build_bundle(features_path, meta_path, graph_path, edges_path):
    feats = pd.read_parquet(features_path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    num, cat = meta["numeric_features"], meta["categorical_features"]

    # keep only usable rows, sort by time (so split blocks are contiguous + history is causal)
    df = feats[feats["split"].isin(["train", "val", "test"])].copy()
    df = df.sort_values("start_datetime", kind="stable").reset_index(drop=True)
    n = len(df)

    split = df["split"].map(SPLIT_CODE).to_numpy().astype(np.int8)
    is_train = split == 0

    # numeric: z-score on TRAIN rows only
    Xn = df[num].to_numpy(dtype=np.float64)
    mu = Xn[is_train].mean(0)
    sd = Xn[is_train].std(0); sd[sd == 0] = 1.0
    Xn = ((Xn - mu) / sd).astype(np.float32)

    Xc = df[cat].to_numpy(dtype=np.int64)
    cat_card = [int(len(meta["category_mappings"][c.replace("_code", "")])) for c in cat]

    node_id = df["node_id"].to_numpy(dtype=np.int64)
    durations = df["duration_min"].to_numpy(dtype=np.float32)
    events = df["event_observed"].to_numpy(dtype=np.int64)
    road_closure = df["road_closure"].to_numpy(dtype=np.int64)
    priority_high = df["priority_high"].to_numpy(dtype=np.int64)
    event_id = df["id"].to_numpy().astype("U16")

    # DeepHit bins from TRAIN uncensored durations
    train_unc = durations[is_train & (events == 1)]
    cuts = make_cuts(train_unc, N_BINS)
    bin_idx = to_bins(durations, cuts)

    # causal history sequences
    hist_idx = build_history(node_id, n, HIST_LEN)

    # Tier-1 point-process targets: time to next incident at the same junction
    tte_min, tte_observed = build_pointprocess_targets(df)

    # graph
    gz = np.load(graph_path)
    edge_index = gz["edge_index"].astype(np.int64)
    num_nodes = int(gz["node_id"].max()) + 1

    node_feat, node_cols = build_node_features(df[is_train], num_nodes, edges_path)

    bundle = dict(
        X_num=Xn, X_cat=Xc, node_id=node_id, hist_idx=hist_idx,
        node_feat=node_feat, edge_index=edge_index,
        durations=durations, events=events, bin_idx=bin_idx, cuts=cuts,
        road_closure=road_closure, priority_high=priority_high,
        tte_min=tte_min, tte_observed=tte_observed,
        split=split, event_id=event_id,
    )
    bundle_meta = {
        "n_rows": n, "num_nodes": num_nodes,
        "numeric_features": num, "categorical_features": cat, "cat_cardinalities": cat_card,
        "node_feature_cols": node_cols,
        "hist_len": HIST_LEN, "n_bins": int(len(cuts) - 1),
        "cuts_min": cuts.tolist(),
        "num_mean": mu.tolist(), "num_std": sd.tolist(),
        "split_counts": {k: int((split == v).sum()) for k, v in SPLIT_CODE.items()},
        "censored_frac": float((events == 0).mean()),
        "tte_observed_frac": float(tte_observed.mean()),
        "tte_median_min": float(np.median(tte_min[tte_observed == 1])),
    }
    return bundle, bundle_meta


def save_bundle(bundle: dict, bundle_meta: dict, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **bundle)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(bundle_meta, indent=2), encoding="utf-8")
    logger.info("Saved bundle -> %s", out_path)
    logger.info("Saved bundle meta -> %s", meta_path)


def _summary(bundle: dict, bm: dict) -> None:
    logger.info("=" * 64)
    logger.info("TRAIN BUNDLE SUMMARY")
    logger.info("  rows .................... %d", bm["n_rows"])
    logger.info("  graph nodes ............. %d", bm["num_nodes"])
    logger.info("  numeric / categorical ... %d / %d (cards=%s)",
                len(bm["numeric_features"]), len(bm["categorical_features"]), bm["cat_cardinalities"])
    logger.info("  node features ........... %d %s", len(bm["node_feature_cols"]), bm["node_feature_cols"])
    logger.info("  history length (L) ...... %d", bm["hist_len"])
    logger.info("  DeepHit bins (K) ........ %d  (cuts %.0f .. %.0f min)",
                bm["n_bins"], bm["cuts_min"][0], bm["cuts_min"][-1])
    logger.info("  split counts ............ %s", bm["split_counts"])
    logger.info("  censored fraction ....... %.3f", bm["censored_frac"])
    hist = bundle["hist_idx"]
    logger.info("  events with >=1 history . %.1f%%", 100 * (hist >= 0).any(1).mean())
    logger.info("  next-event observed ..... %.1f%% (median tte %.0f min)",
                100 * bm["tte_observed_frac"], bm["tte_median_min"])
    logger.info("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Assemble the CASCADE Colab training bundle.")
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--meta", default="data/processed/feature_meta.json")
    ap.add_argument("--graph", default="data/processed/graph.npz")
    ap.add_argument("--edges", default="data/processed/graph_edges.parquet")
    ap.add_argument("--out", default="data/processed/train_bundle.npz")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _setup_logging(verbose=not args.quiet)
    for pth in (args.features, args.meta, args.graph, args.edges):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Missing input: {pth} (run ingest -> graph -> features first)")

    bundle, bm = build_bundle(args.features, args.meta, args.graph, args.edges)
    save_bundle(bundle, bm, args.out)
    _summary(bundle, bm)


if __name__ == "__main__":
    main()
