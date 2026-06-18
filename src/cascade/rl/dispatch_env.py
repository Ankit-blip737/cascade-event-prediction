"""
dispatch_env.py — CASCADE Tier 2/3: the dispatch environment for the RL dispatcher (GATE-PPO).

A fast, numpy-only simulation of a day of incident response on the junction graph, so the policy can
be trained AND validated locally (no GPU needed for the env). The agent commands a small fleet of
response units (BTP's real scarce resource: ~10 tow trucks); it must dispatch them across junctions
to relieve as much congestion as possible, trading travel time against where incidents are — and
where the Hawkes head says the next ones will fire.

State (per junction): active unresolved severity, predicted arrival intensity, distance to the
nearest free unit, plus global time-of-day / free-unit count. Action: which junction to send the
next free unit to (or HOLD to let time advance). Reward: congestion-minutes relieved minus a travel
penalty. Incidents also clear naturally (decay), so acting early captures more — the policy races the
clock and the arrival process.

This module is pure numpy; the policy/PPO live in gate_ppo.py. Greedy/random baselines here give the
number the learned policy must beat.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.rl.env")
EARTH_KM = 6371.0088


def _haversine(lat, lon):
    la = np.radians(lat)[:, None]; lo = np.radians(lon)[:, None]
    dlat = la - la.T; dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class DispatchEnv:
    HOLD = -1  # action index meaning "advance time"

    def __init__(self, lat, lon, intensity, sev_pool, n_units=10, horizon_h=24.0, dt_h=1.0,
                 gain=0.5, travel_penalty=40.0, decay_per_h=0.15, speed_kmph=25.0,
                 service_h=0.75, arrival_scale=1.0, seed=0):
        self.V = len(lat)
        self.D = _haversine(np.asarray(lat, float), np.asarray(lon, float))
        self.intensity = np.maximum(intensity, 1e-6) * arrival_scale   # arrivals/junction/hour
        self.sev_pool = np.asarray(sev_pool, float)
        self.n_units = n_units; self.horizon = horizon_h; self.dt = dt_h
        self.gain = gain; self.travel_penalty = travel_penalty
        self.decay = decay_per_h; self.speed = speed_kmph; self.service_h = service_h
        self.rng = np.random.default_rng(seed)
        self.reset()

    # --- core ---------------------------------------------------------------
    def reset(self):
        self.t = 0.0
        self.active = np.zeros(self.V)
        # units start at the highest-intensity junctions (a sensible depot prior)
        start = np.argsort(-self.intensity)[: self.n_units]
        self.unit_pos = start.copy()
        self.unit_free_at = np.zeros(self.n_units)     # time each unit becomes free
        self._seed_arrivals(self.dt)                   # some incidents already present
        return self.state()

    def _free_units(self):
        return np.where(self.unit_free_at <= self.t + 1e-9)[0]

    def _seed_arrivals(self, dt):
        lam = self.intensity * dt
        counts = self.rng.poisson(lam)
        for j in np.where(counts > 0)[0]:
            self.active[j] += self.sev_pool[self.rng.integers(0, len(self.sev_pool), counts[j])].sum()

    def state(self):
        free = self._free_units()
        if len(free):
            dmin = self.D[:, self.unit_pos[free]].min(axis=1)
        else:
            dmin = np.full(self.V, self.D.max())
        feat = np.stack([
            np.log1p(self.active) / 6.0,                       # current congestion pool
            self.intensity / (self.intensity.max() + 1e-9),    # predicted arrivals
            1.0 - np.clip(dmin / 15.0, 0, 1),                  # proximity to a free unit
            np.full(self.V, len(free) / self.n_units),         # global: free-unit fraction
            np.full(self.V, self.t / self.horizon),            # global: time of day
        ], axis=1).astype(np.float32)                          # [V, 5]
        mask = (self.active > 1e-6) & (len(free) > 0)          # valid dispatch targets
        return {"feat": feat, "mask": mask.astype(bool), "n_free": len(free)}

    def step(self, action):
        free = self._free_units()
        reward = 0.0
        # advance time if HOLD, no free unit, or an invalid/empty target
        if action == self.HOLD or len(free) == 0 or action < 0 or action >= self.V \
                or self.active[action] <= 1e-6:
            self.active *= np.exp(-self.decay * self.dt)       # natural clearance
            self.t += self.dt
            self._seed_arrivals(self.dt)
        else:
            j = int(action)
            u = free[np.argmin(self.D[self.unit_pos[free], j])]  # nearest free unit
            travel_h = self.D[self.unit_pos[u], j] / self.speed
            relieved = self.active[j] * self.gain
            self.active[j] -= relieved
            reward = relieved - self.travel_penalty * travel_h
            self.unit_pos[u] = j
            self.unit_free_at[u] = self.t + travel_h + self.service_h
        done = self.t >= self.horizon - 1e-9
        return self.state(), float(reward), done, {}


# --- baselines (the numbers to beat) -----------------------------------------
def run_policy(env, choose, episodes=20):
    """Average total reward over episodes; `choose(state)->action` is the policy."""
    tot = []
    for _ in range(episodes):
        s = env.reset(); done = False; r_sum = 0.0; steps = 0
        while not done and steps < 5000:
            a = choose(s); s, r, done, _ = env.step(a); r_sum += r; steps += 1
        tot.append(r_sum)
    return float(np.mean(tot)), float(np.std(tot))


def greedy_severity(state):
    if not state["mask"].any():
        return DispatchEnv.HOLD
    masked = np.where(state["mask"], state["feat"][:, 0], -1)
    return int(np.argmax(masked))


def greedy_nearest(state):
    if not state["mask"].any():
        return DispatchEnv.HOLD
    masked = np.where(state["mask"], state["feat"][:, 2], -1)   # proximity feature
    return int(np.argmax(masked))


def random_policy(state, rng=np.random.default_rng(0)):
    if not state["mask"].any():
        return DispatchEnv.HOLD
    return int(rng.choice(np.where(state["mask"])[0]))


# --- factory from artifacts ---------------------------------------------------
def make_env_from_artifacts(calibrated="models/calibrated.npz", nodes="data/processed/graph_nodes.parquet",
                            **kw):
    cal = np.load(calibrated, allow_pickle=True)
    nd = pd.read_parquet(nodes).sort_values("node_id")
    lat, lon = nd["lat"].to_numpy(), nd["lon"].to_numpy()
    intensity = cal["node_intensity"] if "node_intensity" in cal.files else np.full(len(lat), 0.02)
    # incident severity pool = capped clearance minutes, up-weighted where a closure is likely
    sev = np.minimum(cal["median"], 180.0) * (1.0 + 2.0 * cal["closure_prob"])
    sev_pool = sev[sev > 1.0]
    if len(sev_pool) < 10:
        sev_pool = np.array([30.0, 60.0, 120.0])
    return DispatchEnv(lat, lon, intensity, sev_pool, **kw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    env = make_env_from_artifacts(seed=0)
    logger.info("env: V=%d  units=%d  horizon=%.0fh  sev_pool=%d (mean %.0f)",
                env.V, env.n_units, env.horizon, len(env.sev_pool), env.sev_pool.mean())
    for name, pol in [("random", random_policy), ("greedy-nearest", greedy_nearest),
                      ("greedy-severity", greedy_severity)]:
        m, s = run_policy(env, pol, episodes=30)
        logger.info("  %-16s mean reward %.0f +/- %.0f", name, m, s)
