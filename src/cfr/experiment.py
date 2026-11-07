"""Helpers shared by the experiment scripts."""
import numpy as np
from .train import fit, predict
from .metrics import sqrt_pehe, eps_ate


def nn_proxy(x, t, y, tau_hat, idx_eval, idx_pool):
    """Validation proxy for PEHE without counterfactuals: impute each evaluation unit's counterfactual outcome with
    the factual outcome of its nearest neighbour (in x) of the opposite arm within `idx_pool`, and compare the
    estimated effect with the imputed one. Our choice (the paper's procedure is in its supplement, not in the PDF)."""
    errs = []
    for i in idx_eval:
        pool = idx_pool[t[idx_pool] != t[i]]
        j = pool[np.argmin(((x[pool] - x[i]) ** 2).sum(1))]
        tau_nn = (y[i] - y[j]) if t[i] == 1 else (y[j] - y[i])
        errs.append((tau_hat[i] - tau_nn) ** 2)
    return float(np.mean(errs))


def run_ihdp(rep, sp, *, ipm="none", alpha=0.0, seed=0, iters=3000, **kw):
    """Train on sp['tr'], select iteration on sp['va'], report PEHE/ATE errors within-sample (tr+va) and out-of-sample."""
    x, t, y = rep["x"], rep["t"], rep["y"]
    tau = rep["mu1"] - rep["mu0"]
    tr, va, te = sp["tr"], sp["va"], sp["te"]
    net, info = fit(x[tr], t[tr], y[tr], x[va], t[va], y[va], ipm=ipm, alpha=alpha, seed=seed, iters=iters, **kw)
    f0, f1 = predict(net, x, info)
    th = f1 - f0
    wi = np.concatenate([tr, va])
    out = {"pehe_in": sqrt_pehe(th[wi], tau[wi]), "ate_in": eps_ate(th[wi], tau[wi]),
           "pehe_out": sqrt_pehe(th[te], tau[te]), "ate_out": eps_ate(th[te], tau[te]),
           "val_loss": info["val_loss"], "nn_proxy": nn_proxy(x, t, y, th, va, wi)}
    return out
