"""
ddpm.py — CASCADE Tier 3: conditional diffusion for rare-event augmentation + uncertainty.

Planned events are only ~6% of the ASTRAM log (467 / 8173), so models starve on them. A small
class-conditional DDPM (denoising diffusion) learns the incident feature distribution and GENERATES
synthetic planned-event feature vectors to augment the minority class. The same generative model
gives a sampling-based uncertainty handle (draw many scenarios).

Light MLP epsilon-predictor over the standardized feature space — trains on CPU in well under a
minute, so it is built and validated locally (no Colab). We check the synthetic samples are
realistic two ways: a real-vs-synthetic classifier AUC (near 0.5 == indistinguishable) and per-
feature mean alignment.

Usage:
    python -m src.cascade.diffusion.ddpm --epochs 400 --n-synth 2000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

# Import torch BEFORE pandas/pyarrow: on Windows, loading pyarrow first breaks torch's c10.dll init
# (the same OpenMP/DLL load-order conflict that segfaults OR-Tools).
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

logger = logging.getLogger("cascade.diffusion")
COND_COL = "event_type_planned"   # class condition: 1 = planned (rare)


def _setup_logging(v=True):
    logging.basicConfig(level=logging.INFO if v else logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")


def _time_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class EpsNet(nn.Module):
    """Predict the noise added to x_t, conditioned on diffusion step t and class c."""
    def __init__(self, d, temb=32, hidden=256):
        super().__init__()
        self.temb = temb
        self.cls = nn.Embedding(2, temb)
        self.net = nn.Sequential(
            nn.Linear(d + 2 * temb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d))

    def forward(self, x, t, c):
        h = torch.cat([x, _time_emb(t, self.temb), self.cls(c)], dim=1)
        return self.net(h)


class DDPM:
    def __init__(self, d, T=100, device="cpu"):
        self.T, self.device, self.d = T, device, d
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, 0)
        self.net = EpsNet(d).to(device)

    def q_sample(self, x0, t, noise):
        ab = self.abar[t][:, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, x0, c):
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.net(self.q_sample(x0, t, noise), t, c), noise)

    @torch.no_grad()
    def sample(self, n, c_value):
        x = torch.randn(n, self.d, device=self.device)
        c = torch.full((n,), int(c_value), dtype=torch.long, device=self.device)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.net(x, tt, c)
            a, ab, b = self.alpha[t], self.abar[t], self.beta[t]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + (b.sqrt() * torch.randn_like(x) if t > 0 else 0.0)
        return x


def real_vs_synth_auc(real, synth):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.vstack([real, synth]); y = np.r_[np.ones(len(real)), np.zeros(len(synth))]
    p = LogisticRegression(max_iter=500).fit(X, y).predict_proba(X)[:, 1]
    return float(roc_auc_score(y, p))


def run(features_path, meta_path, epochs, n_synth, out_path):
    feats = pd.read_parquet(features_path); meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    num = [c for c in meta["numeric_features"] if c != COND_COL]
    train = feats[feats["split"] == "train"]
    Xall = train[num].to_numpy(np.float64)
    mu, sd = Xall.mean(0), Xall.std(0); sd[sd == 0] = 1.0
    X = ((Xall - mu) / sd).astype(np.float32)
    c = train[COND_COL].to_numpy().astype(np.int64)
    logger.info("train rows %d | features %d | planned (rare) %d (%.1f%%)",
                len(X), len(num), int(c.sum()), 100 * c.mean())

    torch.manual_seed(0)
    ddpm = DDPM(d=X.shape[1], T=100)
    opt = torch.optim.Adam(ddpm.net.parameters(), lr=2e-3)
    Xt = torch.as_tensor(X); ct = torch.as_tensor(c); n = len(X); bs = 512
    for ep in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]; opt.zero_grad()
            loss = ddpm.loss(Xt[b], ct[b]); loss.backward(); opt.step()
        if ep % max(1, epochs // 5) == 0:
            logger.info("  epoch %3d  ddpm loss %.4f", ep, loss.item())

    synth = ddpm.sample(n_synth, c_value=1).cpu().numpy()              # synthetic PLANNED events
    real_planned = X[c == 1]
    auc = real_vs_synth_auc(real_planned, synth[: len(real_planned) * 3])
    mean_gap = float(np.mean(np.abs(synth.mean(0) - real_planned.mean(0))))

    logger.info("=" * 60)
    logger.info("DIFFUSION AUGMENTATION (planned-event class)")
    logger.info("  real planned (train) ........ %d", len(real_planned))
    logger.info("  synthetic generated ......... %d", n_synth)
    logger.info("  real-vs-synth AUC ........... %.3f  (0.5 = indistinguishable; lower is better)", auc)
    logger.info("  mean |feature-mean gap| ..... %.3f std-units", mean_gap)
    logger.info("=" * 60)

    # de-standardize synthetic features back to real units for downstream use
    synth_real = synth * sd + mu
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, synth_std=synth, synth_real=synth_real,
                        feature_names=np.array(num), mu=mu, sd=sd, cond="planned")
    report = {"real_planned": int(len(real_planned)), "n_synth": int(n_synth),
              "real_vs_synth_auc": round(auc, 3), "mean_feature_gap_std": round(mean_gap, 3)}
    Path("models/diffusion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(ddpm.net.state_dict(), "models/ddpm.pt")
    logger.info("Saved synthetic events -> %s , model -> models/ddpm.pt", out_path)
    return report


def main():
    ap = argparse.ArgumentParser(description="Conditional DDPM for rare planned-event augmentation.")
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--meta", default="data/processed/feature_meta.json")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--out", default="models/synth_planned.npz")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _setup_logging(not args.quiet)
    run(args.features, args.meta, args.epochs, args.n_synth, args.out)


if __name__ == "__main__":
    main()
