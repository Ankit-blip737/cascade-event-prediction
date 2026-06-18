"""
mappls.py — OPTIONAL Mappls (MapmyIndia) service client. ADDITIVE: never replaces the offline path.

Mappls is the sponsor's mapping service (officially provided, scores sponsor points) but its APIs are
rate-limited and can lag, so this client is a *progressive enhancement*: every function returns `None`
on any problem (no key, no `requests`, network error, timeout, bad response) and the caller falls back
to the self-contained haversine/networkx implementation. Mappls is NEVER a hard dependency, and it
NEVER touches modeling — only geometry/routing at the decision/visualization layer.

Enable by setting an env var with your Mappls REST key:  MAPPLS_KEY=...  (or MAPMYINDIA_KEY=...).
Endpoints follow the Mappls Advanced Maps REST style; adjust the URL/auth to match your Mappls plan
(some accounts use OAuth tokens instead of a URL key). If unset, everything transparently uses offline.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("cascade.geo.mappls")

TIMEOUT_S = 4.0          # short: APIs can lag -> fail fast and fall back, never block the demo
_BASE = "https://apis.mappls.com/advancedmaps/v1"


def _key() -> str | None:
    return os.environ.get("MAPPLS_KEY") or os.environ.get("MAPMYINDIA_KEY")


def enabled() -> bool:
    """True only if a key is set AND `requests` is importable. Otherwise callers use the offline path."""
    if not _key():
        return False
    try:
        import requests  # noqa: F401
        return True
    except Exception:
        return False


def _get(url: str, params: dict | None = None):
    import requests
    r = requests.get(url, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def route(coords: list[tuple[float, float]]) -> dict | None:
    """
    Real drive route through `coords` (each (lat, lon)). Returns
    {distance_km, duration_min, geometry:[(lat,lon),...]} or None (-> caller falls back to haversine).
    """
    if not enabled() or len(coords) < 2:
        return None
    try:
        pts = ";".join(f"{lon},{lat}" for lat, lon in coords)        # Mappls wants lng,lat
        url = f"{_BASE}/{_key()}/route_adv/driving/{pts}"
        j = _get(url, params={"geometries": "geojson", "overview": "full"})
        rt = j["routes"][0]
        geom = [(c[1], c[0]) for c in rt["geometry"]["coordinates"]]  # back to (lat,lon)
        return {"distance_km": rt["distance"] / 1000.0,
                "duration_min": rt["duration"] / 60.0, "geometry": geom}
    except Exception as e:  # noqa: BLE001 — any failure -> offline fallback
        logger.info("Mappls route unavailable (%s); using offline fallback.", type(e).__name__)
        return None


def snap_to_road(coords: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """Snap points to the road network. Returns snapped [(lat,lon),...] or None (-> keep raw coords)."""
    if not enabled() or not coords:
        return None
    try:
        pts = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url = f"{_BASE}/{_key()}/snapToRoad"
        j = _get(url, params={"path": pts})
        sp = j.get("snappedPoints") or j.get("results") or []
        out = [(p["location"]["latitude"], p["location"]["longitude"]) for p in sp]
        return out or None
    except Exception as e:  # noqa: BLE001
        logger.info("Mappls snap-to-road unavailable (%s); keeping raw coords.", type(e).__name__)
        return None


def drive_distance_km(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    """Real driving distance a->b in km, or None (-> caller uses haversine)."""
    r = route([a, b])
    return r["distance_km"] if r else None
