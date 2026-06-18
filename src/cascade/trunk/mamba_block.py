"""
mamba_block.py — CASCADE Tier-1 swap: a selective-SSM (Mamba) temporal kernel, GUARDED.

Drop-in replacement for the trunk's GRU temporal kernel. Per CLAUDE.md the GRU is the safe default
and `mamba-ssm`'s custom CUDA kernels are an install trap, so this ships a **pure-PyTorch** selective
scan (Mamba / S6) that needs no kernel compilation: it runs on CPU, is testable locally, and the
architecture's identity does not depend on the optimized kernel.

Selectivity: Delta, B, C are input-dependent (the "selective" mechanism); the state matrix A is
diagonal and parameterized as A = -exp(A_log) so its real part is always negative -> the recurrence
is stable (CLAUDE.md patch #2: bounded Delta / negative-real A). Our history sequences are short
(L=8), so the O(L) sequential scan is cheap even unfused.

`MambaTemporal` exposes the same contract the trunk needs: a sequence [B, L, d] -> a single summary
vector [B, d] (the last selective-scan step), exactly replacing `_, hn = gru(seq); hn.squeeze(0)`.

If the optimized `mamba_ssm` package is importable it could be swapped in, but the pure-torch path is
the portable default used everywhere here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mamba_ssm_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        return True
    except Exception:
        return False


class MambaBlock(nn.Module):
    """Minimal Mamba (S6) block in pure PyTorch: conv -> selective SSM -> gated output."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)          # -> x stream + gate z
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)   # causal depthwise
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)  # selective Delta,B,C
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # A = -exp(A_log) (diagonal, negative-real => stable). Init S4D-style.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))             # skip connection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # x [B, L, d_model]
        B, L, _ = x.shape
        xz = self.in_proj(x)                                        # [B, L, 2*d_inner]
        xin, z = xz.chunk(2, dim=-1)                                # each [B, L, d_inner]

        # causal depthwise conv over time, then SiLU
        xc = self.conv1d(xin.transpose(1, 2))[..., :L].transpose(1, 2)
        xin = F.silu(xc)

        # input-dependent (selective) parameters
        dbc = self.x_proj(xin)                                      # [B, L, dt_rank + 2*d_state]
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).clamp(max=10.0)        # [B, L, d_inner], bounded
        A = -torch.exp(self.A_log)                                  # [d_inner, d_state], stable

        # discretize (zero-order hold): dA = exp(delta*A), dB = delta*B
        dA = torch.exp(delta.unsqueeze(-1) * A)                     # [B, L, d_inner, d_state]
        dBx = delta.unsqueeze(-1) * Bm.unsqueeze(2) * xin.unsqueeze(-1)  # [B, L, d_inner, d_state]

        # sequential selective scan (L is small)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))         # [B, d_inner]
        y = torch.stack(ys, dim=1) + xin * self.D                  # [B, L, d_inner]
        y = y * F.silu(z)                                          # gating
        return self.out_proj(y)                                    # [B, L, d_model]


class MambaTemporal(nn.Module):
    """Temporal kernel with the GRU's contract: sequence [B,L,d] -> summary [B,d] (last step)."""

    def __init__(self, d_model: int, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = MambaBlock(d_model, **kw)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        y = self.block(self.norm(seq)) + seq                       # residual
        return y[:, -1, :]                                         # final selective-scan state


if __name__ == "__main__":
    import logging, time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("mamba")
    torch.manual_seed(0)
    d, L, B = 64, 8, 256
    log.info("mamba_ssm available: %s (using pure-torch path)", mamba_ssm_available())

    # --- shape / drop-in check vs GRU ---
    mamba = MambaTemporal(d); gru = nn.GRU(d, d, batch_first=True)
    seq = torch.randn(B, L, d)
    out_m = mamba(seq); _, hn = gru(seq); out_g = hn.squeeze(0)
    log.info("output shapes  mamba %s  gru %s  (match: %s)", tuple(out_m.shape), tuple(out_g.shape),
             out_m.shape == out_g.shape)
    log.info("params  mamba %d  gru %d", sum(p.numel() for p in mamba.parameters()),
             sum(p.numel() for p in gru.parameters()))

    # --- selectivity test: target depends on a value at a CUED earlier timestep ---
    # each sequence: feature 0 is a cue position (one-hot over L in dims 1..L), value in feature 0;
    # target = the value at the cued timestep -> the kernel must selectively remember it.
    def make_batch(n):
        x = torch.randn(n, L, d) * 0.1
        cue = torch.randint(0, L, (n,))
        vals = torch.randn(n)
        for i in range(n):
            x[i, cue[i], 0] = vals[i]
            x[i, :, 1] = 0; x[i, cue[i], 1] = 1.0                  # cue marker channel
        return x, vals
    for name, kernel, head in [("Mamba", MambaTemporal(d), nn.Linear(d, 1)),
                               ("GRU", None, None)]:
        if name == "GRU":
            kernel = nn.GRU(d, d, batch_first=True); head = nn.Linear(d, 1)
            fwd = lambda s: head(kernel(s)[1].squeeze(0))
        else:
            fwd = lambda s: head(kernel(s))
        opt = torch.optim.Adam(list(kernel.parameters()) + list(head.parameters()), lr=3e-3)
        t0 = time.time()
        for step in range(400):
            x, y = make_batch(256); opt.zero_grad()
            loss = F.mse_loss(fwd(x).squeeze(-1), y); loss.backward(); opt.step()
        xv, yv = make_batch(1000)
        with torch.no_grad():
            mse = F.mse_loss(fwd(xv).squeeze(-1), yv).item()
        log.info("  %-6s selective-copy test MSE %.4f  (%.1fs)", name, mse, time.time() - t0)
