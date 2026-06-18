"""
gate_ppo.py — CASCADE Tier 2/3: the GATE-PPO dispatcher (graph-encoder PPO).

A graph-encoder policy trained with PPO to dispatch the response fleet on the junction graph
(dispatch_env.py). The encoder is a 2-layer graph convolution over the junctions (a GAT head is a
drop-in); the actor scores every junction for "send the next free unit here" plus a HOLD action, and
the critic reads a pooled graph embedding. Trained to beat the greedy-severity baseline by trading
travel against congestion and pre-positioning toward where the Hawkes head predicts arrivals.

Light enough to train on CPU locally (small net + fast numpy env), so it is validated here, not blind
on Colab. Saves the policy + a reward comparison to models/.

Usage:
    python -m src.cascade.rl.gate_ppo --updates 150 --out models/dispatcher.pt
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cascade.rl.dispatch_env import (make_env_from_artifacts, run_policy,
                                         greedy_severity, greedy_nearest, random_policy, DispatchEnv)

logger = logging.getLogger("cascade.rl.ppo")
REWARD_SCALE = 0.01    # keep PPO targets ~O(1)


def _setup_logging(v=True):
    logging.basicConfig(level=logging.INFO if v else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")


def norm_adj_dense(edge_index, V):
    A = np.eye(V, dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    d = A.sum(1); dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    return torch.as_tensor((dinv[:, None] * A) * dinv[None, :], dtype=torch.float32)


class GraphPolicy(nn.Module):
    def __init__(self, in_feat, V, adj, hidden=64):
        super().__init__()
        self.register_buffer("adj", adj)
        self.g1 = nn.Linear(in_feat, hidden); self.g2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, 1)
        self.sev_w = nn.Parameter(torch.tensor(3.0))        # warm-start: logit ~ severity (greedy prior)
        self.hold = nn.Parameter(torch.zeros(1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, feat):                                  # [V,in] (single) or [B,V,in] (batched)
        if feat.dim() == 2:
            h = F.relu(self.adj @ self.g1(feat)); return F.relu(self.adj @ self.g2(h))
        h = F.relu(torch.einsum("vw,bwh->bvh", self.adj, self.g1(feat)))
        return F.relu(torch.einsum("vw,bwh->bvh", self.adj, self.g2(h)))

    def forward(self, feat, mask):                           # single state -> logits [V+1], value
        h = self.encode(feat)
        node_logits = self.actor(h).squeeze(-1) + self.sev_w * feat[:, 0]   # +severity skip
        m = torch.as_tensor(mask)
        hold_valid = ~m.any()                               # HOLD only when nothing to dispatch
        logits = torch.cat([node_logits, self.hold])        # [V+1]; last == HOLD
        valid = torch.cat([m, hold_valid.reshape(1)])
        logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
        return logits, self.critic(h.mean(0)).squeeze(-1)

    def eval_actions(self, feat_b, mask_b, acts):           # batched [B,V,in], [B,V] -> logp, value, entropy
        h = self.encode(feat_b)
        node_logits = self.actor(h).squeeze(-1) + self.sev_w * feat_b[:, :, 0]   # +severity skip
        B = node_logits.shape[0]
        hold_valid = ~mask_b.any(1, keepdim=True)           # [B,1]
        logits = torch.cat([node_logits, self.hold.expand(B, 1)], 1)
        valid = torch.cat([mask_b, hold_valid], 1)
        logits = logits.masked_fill(~valid, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.critic(h.mean(1)).squeeze(-1)
        return dist.log_prob(acts), value, dist.entropy()


def act(policy, state):
    feat = torch.as_tensor(state["feat"])
    logits, value = policy(feat, state["mask"])
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return a.item(), dist.log_prob(a), value, dist.entropy()


def to_env_action(a, V):
    return DispatchEnv.HOLD if a == V else a


def collect(policy, env, steps):
    """Run the env under the current policy for ~`steps` transitions; return a trajectory buffer."""
    buf = {k: [] for k in ["feat", "mask", "act", "logp", "val", "rew", "done"]}
    s = env.reset(); ep_ret = 0.0; ep_rets = []
    for _ in range(steps):
        a, logp, val, _ = act(policy, s)
        ns, r, done, _ = env.step(to_env_action(a, env.V))
        buf["feat"].append(s["feat"]); buf["mask"].append(s["mask"]); buf["act"].append(a)
        buf["logp"].append(logp.detach()); buf["val"].append(val.detach())
        buf["rew"].append(r * REWARD_SCALE); buf["done"].append(done)
        ep_ret += r; s = ns
        if done:
            ep_rets.append(ep_ret); ep_ret = 0.0; s = env.reset()
    return buf, (np.mean(ep_rets) if ep_rets else ep_ret)


def gae(rew, val, done, gamma=0.99, lam=0.95):
    adv = np.zeros(len(rew), dtype=np.float32); last = 0.0
    for t in reversed(range(len(rew))):
        nonterm = 1.0 - float(done[t])
        nextv = val[t + 1] if t + 1 < len(rew) else 0.0
        delta = rew[t] + gamma * nextv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + np.asarray(val, dtype=np.float32)


def ppo_update(policy, opt, buf, epochs=4, clip=0.2, vf=0.5, ent=0.01):
    val = np.asarray([v.item() for v in buf["val"]], dtype=np.float32)
    adv, ret = gae(buf["rew"], val, buf["done"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    feat_b = torch.as_tensor(np.stack(buf["feat"]))                 # [N,V,in]
    mask_b = torch.as_tensor(np.stack(buf["mask"]))                 # [N,V] bool
    acts = torch.as_tensor(buf["act"]); old_logp = torch.stack(buf["logp"])
    adv_t = torch.as_tensor(adv); ret_t = torch.as_tensor(ret)
    n = len(acts); idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s0 in range(0, n, 256):
            mb = idx[s0:s0 + 256]
            logp, v, e = policy.eval_actions(feat_b[mb], mask_b[mb], acts[mb])
            ratio = torch.exp(logp - old_logp[mb])
            s1 = ratio * adv_t[mb]; s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
            loss = -torch.min(s1, s2).mean() + vf * F.mse_loss(v, ret_t[mb]) - ent * e.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()


def train(env, edge_index, updates=150, rollout=1024, lr=3e-4, seed=0, out_path="models/dispatcher.pt"):
    torch.manual_seed(seed); np.random.seed(seed)
    adj = norm_adj_dense(edge_index, env.V)
    policy = GraphPolicy(in_feat=env.state()["feat"].shape[1], V=env.V, adj=adj)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    g_sev, _ = run_policy(env, greedy_severity, episodes=20)
    logger.info("baseline greedy-severity reward = %.0f", g_sev)

    best = -1e18
    for u in range(updates):
        buf, ep_ret = collect(policy, env, rollout)
        ppo_update(policy, opt, buf)
        if u % 15 == 0 or u == updates - 1:
            m, _ = run_policy(env, lambda s: _greedy_from_policy(policy, s), episodes=15)
            logger.info("update %3d  train_ep_ret %.0f  eval_reward %.0f  (greedy %.0f)", u, ep_ret, m, g_sev)
            if m > best:
                best = m; torch.save({"state": policy.state_dict(), "edge_index": edge_index,
                                      "in_feat": env.state()["feat"].shape[1], "V": env.V}, out_path)
    return policy, best, g_sev


@torch.no_grad()
def _greedy_from_policy(policy, state):
    feat = torch.as_tensor(state["feat"])
    logits, _ = policy(feat, state["mask"])
    return to_env_action(int(torch.argmax(logits).item()), len(state["mask"]))


def run(calibrated, nodes, graph, updates, out_path):
    env = make_env_from_artifacts(calibrated=calibrated, nodes=nodes, seed=0)
    edge_index = np.load(graph)["edge_index"]
    policy, best, g_sev = train(env, edge_index, updates=updates, out_path=out_path)

    # final comparison vs all baselines
    rows = {}
    for nm, pol in [("random", random_policy), ("greedy-nearest", greedy_nearest),
                    ("greedy-severity", greedy_severity),
                    ("GATE-PPO (learned)", lambda s: _greedy_from_policy(policy, s))]:
        m, sd = run_policy(env, pol, episodes=40); rows[nm] = round(m, 0)
    logger.info("=" * 60)
    logger.info("DISPATCHER REWARD (congestion relieved, higher=better)")
    for nm, m in rows.items():
        logger.info("  %-22s %.0f", nm, m)
    lift = 100 * (rows["GATE-PPO (learned)"] - rows["greedy-severity"]) / abs(rows["greedy-severity"])
    logger.info("  GATE-PPO vs greedy-severity: %+.1f%%", lift)
    logger.info("=" * 60)
    Path("models/dispatcher_report.json").write_text(json.dumps(
        {"rewards": rows, "lift_vs_greedy_pct": round(lift, 1), "best_eval": round(best, 0)}, indent=2))
    logger.info("Saved policy -> %s , report -> models/dispatcher_report.json", out_path)


def main():
    ap = argparse.ArgumentParser(description="Train the GATE-PPO dispatcher.")
    ap.add_argument("--calibrated", default="models/calibrated.npz")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.parquet")
    ap.add_argument("--graph", default="data/processed/graph.npz")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--out", default="models/dispatcher.pt")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    run(args.calibrated, args.nodes, args.graph, args.updates, args.out)


if __name__ == "__main__":
    main()
