"""
graph.py — CASCADE Tier 0, Component #2: the road-graph skeleton.

Turns the cleaned ASTRAM events into a *junction graph* — the conveyor-belt skeleton the
GCN trunk runs on: junctions = nodes (stations), roads = edges (belts between them).

Design decisions (driven by the actual data, see CLAUDE.md §2 + EDA):
  * `latitude`/`longitude` are ALWAYS present and clean (no nulls/zeros, all inside Bengaluru).
    `junction` is null on ~69% of rows and `zone` on ~58%, so coordinates are the reliable signal.
  * NODES = the 294 *named* junctions, each given a robust centroid (median lat/long of its
    member events) and a dominant zone. These are the canonical graph nodes.
  * EVERY event — including the ~5,663 with no junction name — is SNAPPED to its nearest
    junction node by great-circle (haversine) distance. Named events keep their ground-truth node.
  * EDGES come from two sources, unioned into one undirected graph:
      1. geographic k-NN  — each junction wired to its k nearest neighbours (haversine),
      2. corridor chains  — junctions sharing a named corridor are linked in geographic order
                            along that corridor's principal axis (injects road-following topology).
    "Non-corridor" is a catch-all bucket, not a real road, so it is excluded from chaining.

OSMnx is intentionally OPTIONAL and OFF by default (heavy OSM download + version traps). The
native k-NN+corridor graph is a faithful, fast skeleton. torch_geometric is also guarded — the
portable `graph.npz` (edge_index + coords) is the handoff to dataset.py / Colab whether or not
PyG is installed locally.

Outputs (all under data/processed/):
  graph_nodes.parquet     one row per junction node (node_id, junction, zone, lat, lon, n_events)
  graph_edges.parquet     undirected edge list (u, v, dist_km, kind)
  graph.npz               edge_index [2, 2E] int64 (both directions), node_lat/lon/id arrays
  events_nodes.parquet    id -> node_id mapping (+ snap distance) for every event

Usage:
    python -m src.cascade.data.graph \
        --input  data/processed/events_clean.parquet \
        --outdir data/processed --k 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.graph")

EARTH_RADIUS_KM = 6371.0088

# Column names in the cleaned events frame.
ID_COL = "id"
JUNCTION_COL = "junction"
ZONE_COL = "zone"
CORRIDOR_COL = "corridor"
LAT_COL = "latitude"
LON_COL = "longitude"

NON_CORRIDOR = "non-corridor"  # excluded from corridor chaining (catch-all, not a road)


# --- helpers ------------------------------------------------------------------
def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _radians(latlon: np.ndarray) -> np.ndarray:
    """[N,2] (lat,lon) degrees -> radians, for sklearn's haversine BallTree."""
    return np.radians(latlon.astype(float))


def _balltree(latlon_deg: np.ndarray):
    """Build a haversine BallTree over (lat,lon) in degrees. Distances come back in radians."""
    from sklearn.neighbors import BallTree  # local import: keep module import light

    return BallTree(_radians(latlon_deg), metric="haversine")


# --- nodes --------------------------------------------------------------------
def build_junction_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    One node per *named* junction. Coordinates = median of member events (outlier-robust);
    zone = the most common zone seen at that junction. Returns a node table indexed 0..N-1.
    """
    named = df[df[JUNCTION_COL].notna()].copy()
    if named.empty:
        raise ValueError("No rows have a non-null 'junction' — cannot build junction nodes.")

    def _mode_or_none(s: pd.Series):
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else None

    grp = named.groupby(JUNCTION_COL)
    nodes = pd.DataFrame({
        "junction": list(grp.groups.keys()),
        "lat": grp[LAT_COL].median().values,
        "lon": grp[LON_COL].median().values,
        "zone": grp[ZONE_COL].apply(_mode_or_none).values,
        "n_events": grp.size().values,
    })
    nodes = nodes.sort_values("junction").reset_index(drop=True)
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    logger.info("Built %d junction nodes from %d named events.", len(nodes), len(named))
    return nodes


def snap_events_to_nodes(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Assign every event a `node_id`:
      * named junctions  -> their own node (ground truth preserved),
      * unnamed events    -> nearest junction node by haversine distance.
    Returns a frame [id, node_id, snap_dist_km, snapped(bool)].
    """
    name_to_id = dict(zip(nodes["junction"], nodes["node_id"]))
    node_tree = _balltree(nodes[["lat", "lon"]].values)

    # nearest node for ALL events (cheap; we only keep it for the unnamed ones)
    dist_rad, idx = node_tree.query(_radians(df[[LAT_COL, LON_COL]].values), k=1)
    nearest_id = nodes["node_id"].values[idx[:, 0]]
    nearest_km = dist_rad[:, 0] * EARTH_RADIUS_KM

    has_name = df[JUNCTION_COL].notna().values
    named_id = df[JUNCTION_COL].map(name_to_id).values  # NaN where unnamed / unknown name

    node_id = np.where(has_name & ~pd.isna(named_id), named_id, nearest_id).astype(np.int64)
    snapped = ~has_name
    snap_km = np.where(snapped, nearest_km, 0.0)

    out = pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "node_id": node_id,
        "snap_dist_km": snap_km,
        "snapped": snapped,
    })
    n_snap = int(snapped.sum())
    logger.info("Snapped %d unnamed events to nearest node (median snap dist %.2f km, p90 %.2f km).",
                n_snap,
                float(np.median(snap_km[snapped])) if n_snap else 0.0,
                float(np.quantile(snap_km[snapped], 0.90)) if n_snap else 0.0)
    return out


# --- edges --------------------------------------------------------------------
def _knn_edges(nodes: pd.DataFrame, k: int) -> set[tuple[int, int]]:
    """Each node -> its k nearest neighbours by haversine. Returned as unordered (u<v) pairs."""
    coords = nodes[["lat", "lon"]].values
    tree = _balltree(coords)
    k_eff = min(k + 1, len(nodes))  # +1 because the nearest hit is the node itself
    _, idx = tree.query(_radians(coords), k=k_eff)
    ids = nodes["node_id"].values
    edges: set[tuple[int, int]] = set()
    for row_pos, neigh in enumerate(idx):
        u = int(ids[row_pos])
        for col in neigh[1:]:  # skip self
            v = int(ids[col])
            if u != v:
                edges.add((min(u, v), max(u, v)))
    logger.info("k-NN (k=%d) produced %d undirected edges.", k, len(edges))
    return edges


def _corridor_edges(df: pd.DataFrame, nodes: pd.DataFrame) -> set[tuple[int, int]]:
    """
    Link junctions that share a named corridor, ordered along the corridor's principal
    geographic axis (1-D PCA on the member-node coords). Consecutive nodes are connected,
    giving the graph road-following structure. 'Non-corridor' is excluded.
    """
    name_to_id = dict(zip(nodes["junction"], nodes["node_id"]))
    id_to_coord = {int(r.node_id): (r.lat, r.lon) for r in nodes.itertuples()}

    named = df[df[JUNCTION_COL].notna() & df[CORRIDOR_COL].notna()].copy()
    named["_corr"] = named[CORRIDOR_COL].astype(str).str.strip().str.lower()
    named = named[named["_corr"] != NON_CORRIDOR]

    edges: set[tuple[int, int]] = set()
    for corr, g in named.groupby("_corr"):
        node_ids = sorted({name_to_id[j] for j in g[JUNCTION_COL].unique() if j in name_to_id})
        if len(node_ids) < 2:
            continue
        pts = np.array([id_to_coord[i] for i in node_ids], dtype=float)
        # project onto principal axis to get a 1-D ordering along the corridor
        centered = pts - pts.mean(axis=0)
        # principal direction via SVD (robust for tiny sets)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        order = np.argsort(centered @ vh[0])
        chain = [node_ids[o] for o in order]
        for a, b in zip(chain[:-1], chain[1:]):
            edges.add((min(a, b), max(a, b)))
    logger.info("Corridor chains produced %d undirected edges across %d corridors.",
                len(edges), named["_corr"].nunique())
    return edges


def build_edges(df: pd.DataFrame, nodes: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Union of geographic k-NN and corridor-chain edges -> undirected edge table with km weights."""
    knn = _knn_edges(nodes, k)
    corr = _corridor_edges(df, nodes)

    id_to_coord = {int(r.node_id): (r.lat, r.lon) for r in nodes.itertuples()}

    def _km(u: int, v: int) -> float:
        (la1, lo1), (la2, lo2) = id_to_coord[u], id_to_coord[v]
        p = np.radians([la1, lo1]); q = np.radians([la2, lo2])
        dlat, dlon = q[0] - p[0], q[1] - p[1]
        a = np.sin(dlat / 2) ** 2 + np.cos(p[0]) * np.cos(q[0]) * np.sin(dlon / 2) ** 2
        return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))

    rows = []
    for (u, v) in sorted(knn | corr):
        kind = "both" if (u, v) in knn and (u, v) in corr else ("knn" if (u, v) in knn else "corridor")
        rows.append((u, v, _km(u, v), kind))
    edges = pd.DataFrame(rows, columns=["u", "v", "dist_km", "kind"])
    logger.info("Total undirected edges: %d (knn=%d, corridor=%d, overlap=%d).",
                len(edges), len(knn), len(corr), len(knn & corr))
    return edges


# --- graph objects ------------------------------------------------------------
def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Build an undirected nx.Graph (node attrs + km edge weights) for routing/diversion later."""
    import networkx as nx

    G = nx.Graph()
    for r in nodes.itertuples():
        G.add_node(int(r.node_id), junction=r.junction, zone=r.zone,
                   lat=float(r.lat), lon=float(r.lon), n_events=int(r.n_events))
    for r in edges.itertuples():
        G.add_edge(int(r.u), int(r.v), dist_km=float(r.dist_km), kind=r.kind)
    n_iso = sum(1 for _, d in G.degree() if d == 0)
    n_comp = nx.number_connected_components(G)
    logger.info("networkx graph: %d nodes, %d edges, %d components, %d isolated.",
                G.number_of_nodes(), G.number_of_edges(), n_comp, n_iso)
    return G


def to_edge_index(edges: pd.DataFrame) -> np.ndarray:
    """Undirected edge table -> directed edge_index [2, 2E] (both directions), PyG convention."""
    u = edges["u"].values
    v = edges["v"].values
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    return np.stack([src, dst]).astype(np.int64)


def to_pyg(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Optional: torch_geometric Data object. Returns None (with a log) if PyG isn't installed."""
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as e:  # noqa: BLE001 — torch is a Colab-only dep locally
        logger.info("torch_geometric not available locally (%s); skipping PyG Data "
                    "(graph.npz is the portable handoff).", type(e).__name__)
        return None

    edge_index = torch.as_tensor(to_edge_index(edges), dtype=torch.long)
    pos = torch.as_tensor(nodes[["lat", "lon"]].values, dtype=torch.float)
    data = Data(edge_index=edge_index, pos=pos, num_nodes=len(nodes))
    logger.info("Built torch_geometric Data: %s", data)
    return data


# --- public API + IO ----------------------------------------------------------
def build_graph(df: pd.DataFrame, k: int = 4):
    """raw cleaned events -> (nodes_df, edges_df, event_node_map_df). The core entry point."""
    nodes = build_junction_nodes(df)
    edges = build_edges(df, nodes, k=k)
    event_nodes = snap_events_to_nodes(df, nodes)
    return nodes, edges, event_nodes


def save_graph(nodes: pd.DataFrame, edges: pd.DataFrame, event_nodes: pd.DataFrame,
               outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    def _pq(df_: pd.DataFrame, name: str) -> None:
        p = outdir / name
        try:
            df_.to_parquet(p, index=False)
            logger.info("Saved %s -> %s", name, p)
        except Exception as e:  # pyarrow missing -> csv fallback
            csv = p.with_suffix(".csv")
            df_.to_csv(csv, index=False)
            logger.warning("Parquet failed for %s (%s); saved CSV -> %s", name, e, csv)

    _pq(nodes, "graph_nodes.parquet")
    _pq(edges, "graph_edges.parquet")
    _pq(event_nodes, "events_nodes.parquet")

    npz = outdir / "graph.npz"
    np.savez(
        npz,
        edge_index=to_edge_index(edges),
        node_id=nodes["node_id"].values.astype(np.int64),
        node_lat=nodes["lat"].values.astype(np.float64),
        node_lon=nodes["lon"].values.astype(np.float64),
    )
    logger.info("Saved portable edge_index/coords -> %s", npz)


def _summary(nodes: pd.DataFrame, edges: pd.DataFrame, event_nodes: pd.DataFrame) -> None:
    deg = pd.concat([edges["u"], edges["v"]]).value_counts()
    logger.info("=" * 60)
    logger.info("GRAPH SUMMARY")
    logger.info("  nodes ................... %d", len(nodes))
    logger.info("  undirected edges ........ %d", len(edges))
    logger.info("  avg degree .............. %.2f", 2 * len(edges) / max(len(nodes), 1))
    logger.info("  isolated nodes .......... %d", len(nodes) - deg.index.nunique())
    logger.info("  events mapped to nodes .. %d", len(event_nodes))
    logger.info("  events snapped (unnamed)  %d (%.1f%%)",
                int(event_nodes["snapped"].sum()),
                100 * event_nodes["snapped"].mean())
    logger.info("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the CASCADE junction graph from cleaned events.")
    ap.add_argument("--input", default="data/processed/events_clean.parquet",
                    help="cleaned events parquet (from ingest.py)")
    ap.add_argument("--outdir", default="data/processed", help="output directory")
    ap.add_argument("--k", type=int, default=4, help="geographic k-NN neighbours per node")
    ap.add_argument("--quiet", action="store_true", help="reduce logging")
    args = ap.parse_args()

    _setup_logging(verbose=not args.quiet)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Cleaned events not found: {in_path} (run ingest.py first)")
    df = pd.read_parquet(in_path)
    logger.info("Loaded %d cleaned events from %s", len(df), in_path)

    nodes, edges, event_nodes = build_graph(df, k=args.k)
    to_networkx(nodes, edges)   # logs connectivity stats; object rebuilt on demand elsewhere
    to_pyg(nodes, edges)        # optional; logs + no-op if PyG missing
    save_graph(nodes, edges, event_nodes, args.outdir)
    _summary(nodes, edges, event_nodes)


if __name__ == "__main__":
    main()
