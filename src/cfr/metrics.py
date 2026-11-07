"""Evaluation metrics of paper Sec 5."""
import numpy as np


def sqrt_pehe(tau_hat, tau):
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def eps_ate(tau_hat, tau):
    return float(abs(np.mean(tau_hat) - np.mean(tau)))


def att_error(f1, f0, t, y, e):
    """eps_ATT = | ATT - (1/|T|) sum_{i in T} (f(x_i,1) - f(x_i,0)) |,
    ATT = |T|^-1 sum_{i in T} y_i - |C cap E|^-1 sum_{i in C cap E} y_i  (E = randomized sample)."""
    T = t == 1
    CE = (t == 0) & (e == 1)
    att = y[T].mean() - y[CE].mean()
    return float(abs(att - (f1[T] - f0[T]).mean()))


def policy_risk(f1, f0, t, y, e, lam=0.0):
    """Empirical policy risk on the randomized part: 1 - (E[Y|pi=1,t=1] p(pi=1) + E[Y|pi=0,t=0] p(pi=0))."""
    r = e == 1
    pi = (f1 - f0) > lam
    pi, tt, yy = pi[r], t[r], y[r]
    p1 = pi.mean()
    a = yy[(pi == 1) & (tt == 1)]
    b = yy[(pi == 0) & (tt == 0)]
    va = a.mean() * p1 if len(a) else 0.0
    vb = b.mean() * (1 - p1) if len(b) else 0.0
    return float(1.0 - (va + vb))


def policy_risk_curve(f1, f0, t, y, e, rates):
    """Policy risk when the top `rate` fraction (by estimated effect) is treated (paper Fig. 3)."""
    r = e == 1
    tau, tt, yy = (f1 - f0)[r], t[r], y[r]
    order = np.argsort(-tau)
    out = []
    n = len(tau)
    for q in rates:
        k = int(round(q * n))
        pi = np.zeros(n, dtype=int); pi[order[:k]] = 1
        p1 = pi.mean()
        a = yy[(pi == 1) & (tt == 1)]; b = yy[(pi == 0) & (tt == 0)]
        out.append(1.0 - ((a.mean() * p1 if len(a) else 0.0) + (b.mean() * (1 - p1) if len(b) else 0.0)))
    return np.array(out)
