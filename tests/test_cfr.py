import math
import numpy as np
import torch
import pytest
from cfr.ipm import mmd2_lin, sinkhorn_wass, sinkhorn_plan, pairwise_dist, exact_wass_1d
from cfr.models import CFRNet, sample_weights
from cfr.metrics import sqrt_pehe, eps_ate, att_error, policy_risk, policy_risk_curve


def test_mmd2_lin_matches_analytic_shift():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(20000, 3, generator=g)
    b = torch.randn(20000, 3, generator=g) + torch.tensor([0.5, 0.0, -0.5])
    assert abs(float(mmd2_lin(a, b)) - 0.5) < 0.03        # ||delta||^2 = 0.25 + 0 + 0.25


def test_sinkhorn_close_to_exact_w1_in_1d():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(400, 1, generator=g)
    b = torch.randn(400, 1, generator=g) + 1.0
    approx = float(sinkhorn_wass(a, b, lam=200.0, iters=200))
    exact = exact_wass_1d(a, b)
    assert abs(approx - exact) / exact < 0.05, (approx, exact)


def test_sinkhorn_plan_has_uniform_marginals_and_zero_distance_is_small():
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(30, 4, generator=g), torch.randn(40, 4, generator=g)
    M = pairwise_dist(a, b)
    P = sinkhorn_plan(M, reg=float(M.mean()) / 10, iters=300)
    assert torch.allclose(P.sum(1), torch.full((30,), 1 / 30), atol=1e-4)
    assert torch.allclose(P.sum(0), torch.full((40,), 1 / 40), atol=1e-4)
    assert float(sinkhorn_wass(a, a.clone(), lam=200.0, iters=200)) < 0.05


def test_wass_envelope_gradient_matches_entropic_ot_gradient():
    """d/dA of the entropic OT value equals <P, dM/dA> (envelope theorem); our loss uses exactly that."""
    torch.manual_seed(3)
    a = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(15, 3, dtype=torch.float64)
    reg, iters = 0.3, 400

    def entropic_value(a_):
        M = pairwise_dist(a_, b)
        P = sinkhorn_plan(M.detach(), reg, iters)
        return (P * M).sum() + reg * (P * (torch.log(P + 1e-300) - 1)).sum()

    v = entropic_value(a); v.backward(); g_env = a.grad.clone()
    num = torch.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            h = 1e-5; ap, am = a.detach().clone(), a.detach().clone(); ap[i, j] += h; am[i, j] -= h
            num[i, j] = (entropic_value(ap) - entropic_value(am)) / (2 * h)
    assert torch.allclose(g_env, num, atol=2e-4), (g_env - num).abs().max()


def test_sample_weights_sum_to_n_and_balance_groups():
    t = torch.tensor([1.0] * 30 + [0.0] * 70)
    w = sample_weights(t)
    assert abs(float(w.sum()) - 100) < 1e-4
    assert abs(float(w[t == 1].sum()) - 50) < 1e-4 and abs(float(w[t == 0].sum()) - 50) < 1e-4


def test_representation_is_unit_norm_and_heads_are_separate():
    net = CFRNet(7)
    x = torch.randn(10, 7)
    r, y0, y1 = net(x)
    assert torch.allclose((r ** 2).sum(1), torch.ones(10), atol=1e-5)
    assert y0.shape == y1.shape == (10,)
    assert not torch.allclose(y0, y1)


def test_metrics():
    th, tau = np.array([1.0, 2, 3]), np.array([1.0, 1, 1])
    assert abs(sqrt_pehe(th, tau) - math.sqrt((0 + 1 + 4) / 3)) < 1e-9
    assert abs(eps_ate(th, tau) - 1.0) < 1e-9


def test_policy_risk_matches_page_toy_example():
    # the 12-person toy of docs/experiments.html: (tau_hat, t, y); k=6 -> R_Pol = 0.25, k=5 -> 0.2208
    toy = [(.9, 1, 0), (.8, 0, 0), (.7, 1, 1), (.6, 1, 1), (.5, 1, 1), (.4, 0, 1), (.3, 1, 1), (.2, 0, 1), (.1, 0, 1), (-.1, 0, 1), (-.3, 1, 0), (-.5, 0, 0)]
    tau, t, y = (np.array(c, dtype=float) for c in zip(*toy))
    e = np.ones(12)
    curve = policy_risk_curve(tau, np.zeros(12), t, y, e, [5 / 12, 6 / 12])
    assert abs(curve[0] - 0.22083) < 1e-4 and abs(curve[1] - 0.25) < 1e-9
    # lam=0 policy treats the 9 people with tau_hat > 0
    assert abs(policy_risk(tau, np.zeros(12), t, y, e, lam=0.0) - policy_risk_curve(tau, np.zeros(12), t, y, e, [9 / 12])[0]) < 1e-9


def test_att_error_on_page_example():
    # treated y = 1,1,0,1,1 (mean .8); randomized controls y = 1,0,1,0 (mean .5) => ATT = .3
    t = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0]); y = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=float); e = np.ones(9)
    f1 = np.r_[np.full(5, 0.5), np.zeros(4)]; f0 = np.zeros(9)
    assert abs(att_error(f1, f0, t, y, e) - 0.2) < 1e-9     # model says 0.5, truth 0.3
