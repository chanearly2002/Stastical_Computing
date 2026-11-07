"""Training loop of Algorithm 1 (mini-batch Adam on Eq. 3) with validation-based model selection."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from .ipm import mmd2_lin, sinkhorn_wass
from .models import CFRNet, sample_weights


def _ipm(kind, r_t, r_c, wass_lam, wass_iters):
    if kind == "mmd":
        return mmd2_lin(r_t, r_c)
    if kind == "wass":
        return sinkhorn_wass(r_t, r_c, lam=wass_lam, iters=wass_iters)
    raise ValueError(kind)


def fit(x, t, y, xv, tv, yv, *, ipm="none", alpha=0.0, loss="mse", iters=3000, bs=100, lr=1e-3,
        decay=0.97, decay_steps=100, lam=1e-4, seed=0, eval_every=100, d_rep=200, n_rep=3, d_head=100,
        n_head=3, wass_lam=10.0, wass_iters=20):
    """Returns (model, info). Inputs are numpy arrays. For loss='mse' the outcome is standardised with the
    training mean/std (info['ym'], info['ys']); for loss='log' y is {0,1} and heads output logits."""
    torch.manual_seed(seed); rs = np.random.RandomState(seed)
    ym, ys = (float(y.mean()), float(y.std() + 1e-8)) if loss == "mse" else (0.0, 1.0)
    X, T, Y = (torch.tensor(a, dtype=torch.float32) for a in (x, t, (y - ym) / ys))
    XV, TV, YV = (torch.tensor(a, dtype=torch.float32) for a in (xv, tv, (yv - ym) / ys))
    W = sample_weights(T)
    WV = TV / (2 * T.mean()) + (1 - TV) / (2 * (1 - T.mean()))      # weights from the training u
    net = CFRNet(X.shape[1], d_rep, n_rep, d_head, n_head)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def factual_loss(yhat0, yhat1, tt, yy, ww):
        pred = tt * yhat1 + (1 - tt) * yhat0
        per = F.binary_cross_entropy_with_logits(pred, yy, reduction="none") if loss == "log" else (pred - yy) ** 2
        return (ww * per).mean()

    best, best_state, hist = float("inf"), None, []
    n = X.shape[0]
    for it in range(1, iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr * decay ** (it // decay_steps)
        idx = torch.tensor(rs.choice(n, size=min(bs, n), replace=False))
        xb, tb, yb, wb = X[idx], T[idx], Y[idx], W[idx]
        rep, y0, y1 = net(xb)
        total = factual_loss(y0, y1, tb, yb, wb) + lam * sum((w ** 2).sum() for w in net.head_weights())
        if alpha > 0 and ipm != "none":
            m1, m0 = tb == 1, tb == 0
            if m1.sum() > 1 and m0.sum() > 1:
                total = total + alpha * _ipm(ipm, rep[m1], rep[m0], wass_lam, wass_iters)
        opt.zero_grad(); total.backward(); opt.step()
        if it % eval_every == 0 or it == iters:
            with torch.no_grad():
                _, v0, v1 = net(XV)
                vl = float(factual_loss(v0, v1, TV, YV, WV))
            hist.append((it, vl))
            if vl < best:
                best, best_state = vl, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return net, {"ym": ym, "ys": ys, "val_loss": best, "hist": hist}


@torch.no_grad()
def predict(net, x, info, loss="mse"):
    """Returns (f(x,0), f(x,1)) in original units (mse) or probabilities (log)."""
    net.eval()
    _, y0, y1 = net(torch.tensor(x, dtype=torch.float32))
    if loss == "log":
        return torch.sigmoid(y0).numpy(), torch.sigmoid(y1).numpy()
    return (y0.numpy() * info["ys"] + info["ym"]), (y1.numpy() * info["ys"] + info["ym"])
