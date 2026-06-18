"""
diversion.py — CASCADE Tier 0/1: the third decision output (after manpower + barricades).

For each barricaded junction, compute the recommended traffic DIVERSION: the best alternate route
between its road-graph neighbours that avoids the barricaded node. Uses the junction graph with
great-circle edge lengths as a travel-cost proxy (Mappls routing can replace this in the dashboard).

Inputs: models/allocation.json (the barricades) + data/processed/graph_{nodes,edges}.parquet.
Output: models/diversions.json — per-barricade reroute (junction sequence + added km).

Usage:
    python -m src.cascade.optimize.diversion
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger("cascade.diversion")


def _setup_logging(verbose=True):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


def build_graph(nodes_path, edges_path):
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    G = nx.Graph()
    for r in nodes.itertuples():
        G.add_node(int(r.node_id), junction=r.junction, lat=float(r.lat), lon=float(r.lon))
    for r in edges.itertuples():
        G.add_edge(int(r.u), int(r.v), w=float(r.dist_km))
    return G, nodes.set_index("node_id")


def divert_around(G, node, name_of):
    """
    Reroute the MAIN corridor through a barricaded node: take its two neighbours spanning the longest
    through-route (the primary road axis), then find the shortest alternate path between them with the
    node removed. Returns the reroute + detour km (negative => the alternate is actually shorter).
    """
    nbrs = list(G.neighbors(node))
    if len(nbrs) < 2:
        return None
    H = G.copy()
    H.remove_node(node)                      # close the barricaded junction
    # primary corridor = neighbour pair with the largest through-junction span
    pair, direct = None, -1.0
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            a, b = nbrs[i], nbrs[j]
            through = G[node][a]["w"] + G[node][b]["w"]
            if through > direct and nx.has_path(H, a, b):
                pair, direct = (a, b), through
    if pair is None:
        return None
    a, b = pair
    alt = nx.shortest_path(H, a, b, weight="w")
    alt_cost = nx.path_weight(H, alt, "w")
    return {"from": name_of(a), "to": name_of(b), "via": [name_of(x) for x in alt],
            "added_km": round(alt_cost - direct, 2), "reroute_km": round(alt_cost, 2),
            # ids retained so an optional Mappls real-route enrichment can use the coordinates
            "from_id": int(a), "to_id": int(b), "via_ids": [int(x) for x in alt]}


def _enrich_with_mappls(d, ntab):
    """ADDITIVE: if Mappls is configured/responsive, attach a real drive route; else leave as-is."""
    try:
        from src.cascade.geo import mappls
    except Exception:
        return d
    if not mappls.enabled():
        return d
    coords = [(float(ntab.loc[i, "lat"]), float(ntab.loc[i, "lon"]))
              for i in d.get("via_ids", []) if i in ntab.index]
    r = mappls.route(coords)
    if r:                                  # offline fields stay; we only ADD the Mappls route
        d["mappls_route"] = {"distance_km": round(r["distance_km"], 2),
                             "duration_min": round(r["duration_min"], 1),
                             "geometry": r["geometry"]}
        d["router_used"] = "mappls"
    else:
        d["router_used"] = "offline"
    return d


def run(allocation_path, nodes_path, edges_path, out_path):
    plan = json.loads(Path(allocation_path).read_text(encoding="utf-8"))
    G, ntab = build_graph(nodes_path, edges_path)
    name_of = lambda v: str(ntab.loc[v, "junction"]) if v in ntab.index else f"node{v}"

    from src.cascade.geo import mappls
    use_mappls = mappls.enabled()
    logger.info("Diversion router: %s (offline haversine/networkx is always the base)",
                "mappls + offline" if use_mappls else "offline only")

    out = []
    for b in plan.get("barricades", []):
        nid = int(b["node_id"])
        if nid not in G:
            continue
        d = divert_around(G, nid, name_of)                 # offline result (always computed)
        if d:
            d["router_used"] = "offline"
            if use_mappls:
                d = _enrich_with_mappls(d, ntab)           # add real route on top, if responsive
            out.append({"barricade": b["junction"], "node_id": nid, **d})

    logger.info("=" * 70)
    logger.info("DIVERSION PLAN  (%d barricades routed)", len(out))
    for d in out[:8]:
        logger.info("    close %-22s  reroute %s -> %s via %d junctions (%+.2f km detour)",
                    d["barricade"], d["from"], d["to"], len(d["via"]), d["added_km"])
    logger.info("=" * 70)

    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Saved diversions -> %s", out_path)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute diversions around barricaded junctions.")
    ap.add_argument("--allocation", default="models/allocation.json")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.parquet")
    ap.add_argument("--edges", default="data/processed/graph_edges.parquet")
    ap.add_argument("--out", default="models/diversions.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    for pth in (args.allocation, args.nodes, args.edges):
        if not Path(pth).exists():
            raise FileNotFoundError(f"Missing {pth}")
    run(args.allocation, args.nodes, args.edges, args.out)


if __name__ == "__main__":
    main()
