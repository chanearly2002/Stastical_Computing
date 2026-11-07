"""Integral probability metrics between two batches of representations (paper Sec 3.2, Algorithm 1 line 6).

mmd2_lin           : squared linear-kernel MMD = || mean(r_t) - mean(r_c) ||^2
sinkhorn_wass      : entropy-regularised Wasserstein-1 (Euclidean ground cost) via Sinkhorn iterations
                     (Cuturi 2013). The transport plan is treated as a constant when differentiating
                     (envelope theorem). The paper's own gradient derivation is in its supplement,
                     which is not part of the PDF, so this is our implementation choice.
exact_wass_1d      : exact 1-D Wasserstein-1 from sorted samples (used for tests only).
"""
import math
import torch


def mmd2_lin(rep_t: torch.Tensor, rep_c: torch.Tensor) -> torch.Tensor:
    return ((rep_t.mean(0) - rep_c.mean(0)) ** 2).sum()


def pairwise_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    d2 = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)
    return torch.sqrt(d2 + eps)


def sinkhorn_plan(M: torch.Tensor, reg: float, iters: int) -> torch.Tensor:
    """Log-domain Sinkhorn for uniform marginals; returns the transport plan (no grad)."""
    n, m = M.shape
    logK = -M / reg
    log_a, log_b = -math.log(n), -math.log(m)
    u = torch.zeros(n, dtype=M.dtype)
    v = torch.zeros(m, dtype=M.dtype)
    for _ in range(iters):
        u = log_a - torch.logsumexp(logK + v[None, :], dim=1)
        v = log_b - torch.logsumexp(logK + u[:, None], dim=0)
    return torch.exp(logK + u[:, None] + v[None, :])


def sinkhorn_wass(rep_t: torch.Tensor, rep_c: torch.Tensor, lam: float = 10.0, iters: int = 20) -> torch.Tensor:
    """<P, M> with P the (detached) Sinkhorn plan; reg = mean(M)/lam as in the public CFR code's convention."""
    M = pairwise_dist(rep_t, rep_c)
    reg = float(M.detach().mean().clamp_min(1e-8)) / lam
    with torch.no_grad():
        P = sinkhorn_plan(M.detach(), reg, iters)
    return (P * M).sum()


def exact_wass_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a.detach().cpu().numpy().ravel(), b.detach().cpu().numpy().ravel()))
