"""Data loading for IHDP (1000 realizations) and Jobs (10 splits), as distributed by the CFR authors.
Files are untrusted data: loaded with allow_pickle=False. Download with scripts/fetch_data.sh."""
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_IHDP = {}


def _ihdp():
    if not _IHDP:
        d = os.path.join(ROOT, "ihdp")
        _IHDP["tr"] = np.load(os.path.join(d, "ihdp_npci_1-1000.train.npz"), allow_pickle=False)
        _IHDP["te"] = np.load(os.path.join(d, "ihdp_npci_1-1000.test.npz"), allow_pickle=False)
        _IHDP["tr"] = {k: _IHDP["tr"][k] for k in _IHDP["tr"].files}
        _IHDP["te"] = {k: _IHDP["te"][k] for k in _IHDP["te"].files}
    return _IHDP["tr"], _IHDP["te"]


def ihdp_rep(r):
    """One outcome realization: all 747 units (672 train + 75 test of the authors' files)."""
    tr, te = _ihdp()
    cat = lambda k: np.concatenate([tr[k][..., r], te[k][..., r]], axis=0)
    return {"x": cat("x"), "t": cat("t"), "y": cat("yf"), "mu0": cat("mu0"), "mu1": cat("mu1"),
            "n_train": tr["t"].shape[0]}


def ihdp_split(rep, seed, val_frac=0.3):
    """Authors' train/test split (672/75); the 672 are split 70/30 into train/val => 63/27/10 overall."""
    n = rep["n_train"]
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    nv = int(round(val_frac * n))
    return {"tr": perm[nv:], "va": perm[:nv], "te": np.arange(n, len(rep["t"]))}


def ihdp_biased(rep, q, seed, n_remove=347):
    """Paper Sec 5.1: fit logistic regression propensity p(t=1|x); repeat n_remove times: with prob q remove the
    remaining control with p-hat closest to 1, otherwise a random control. Returns kept indices (400 units)."""
    from sklearn.linear_model import LogisticRegression
    x, t = rep["x"], rep["t"]
    ph = LogisticRegression(max_iter=2000).fit(x, t).predict_proba(x)[:, 1]
    rs = np.random.RandomState(seed)
    ctrl = list(np.where(t == 0)[0])
    for _ in range(n_remove):
        if rs.rand() < q:
            j = int(np.argmax(ph[ctrl]))
        else:
            j = int(rs.randint(len(ctrl)))
        ctrl.pop(j)
    return np.sort(np.concatenate([np.where(t == 1)[0], np.array(ctrl)]))


def split_indices(n, seed, fracs=(0.63, 0.27, 0.10)):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    a = int(round(fracs[0] * n)); b = a + int(round(fracs[1] * n))
    return {"tr": perm[:a], "va": perm[a:b], "te": perm[b:]}


def jobs_split(i, seed, val_frac=0.3):
    """Jobs: authors' 10 train/test splits (2570/642). The train part is split 70/30 into train/val
    => 56/24/20 overall. x has 17 columns, yf in {0,1} (1 = employed), e = randomized-sample indicator."""
    d = os.path.join(ROOT, "jobs")
    tr = np.load(os.path.join(d, "jobs_DW_bin.new.10.train.npz"), allow_pickle=False)
    te = np.load(os.path.join(d, "jobs_DW_bin.new.10.test.npz"), allow_pickle=False)
    g = lambda z: {"x": z["x"][:, :, i], "t": z["t"][:, i], "y": z["yf"][:, i], "e": z["e"][:, i]}
    a, b = g(tr), g(te)
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(a["t"]))
    nv = int(round(val_frac * len(perm)))
    sub = lambda d_, ix: {k: v[ix] for k, v in d_.items()}
    return sub(a, perm[nv:]), sub(a, perm[:nv]), b
