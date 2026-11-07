"""Simple baselines (a subset of paper Table 1; BART, BNN, TMLE and Causal Forests are not reproduced)."""
import numpy as np


def _fit_ols(X, y):
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _pred_ols(w, X):
    return np.c_[np.ones(len(X)), X] @ w


def ols1(x, t, y, xq):
    """OLS with treatment as a feature."""
    w = _fit_ols(np.c_[x, t], y)
    return _pred_ols(w, np.c_[xq, np.zeros(len(xq))]), _pred_ols(w, np.c_[xq, np.ones(len(xq))])


def ols2(x, t, y, xq):
    """OLS with separate regressors per treatment."""
    w0, w1 = _fit_ols(x[t == 0], y[t == 0]), _fit_ols(x[t == 1], y[t == 1])
    return _pred_ols(w0, xq), _pred_ols(w1, xq)


def knn(x, t, y, xq, k=5):
    """Per-arm k-NN regression (k=5 is our choice; the paper does not state it)."""
    out = []
    for a in (0, 1):
        xa, ya = x[t == a], y[t == a]
        d = ((xq[:, None, :] - xa[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, :k]
        out.append(ya[nn].mean(1))
    return out[0], out[1]


def forest(x, t, y, xq, seed=0, classify=False):
    """Per-arm random forest (our T-learner stand-in for the paper's 'R.For.')."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    out = []
    for a in (0, 1):
        m = (RandomForestClassifier if classify else RandomForestRegressor)(
            n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1)
        m.fit(x[t == a], y[t == a])
        out.append(m.predict_proba(xq)[:, 1] if classify else m.predict(xq))
    return out[0], out[1]


def logreg(x, t, y, xq, separate=False):
    """LR1 (treatment as feature) / LR2 (separate regressors) for binary outcomes."""
    from sklearn.linear_model import LogisticRegression
    if not separate:
        m = LogisticRegression(max_iter=2000).fit(np.c_[x, t], y)
        return m.predict_proba(np.c_[xq, np.zeros(len(xq))])[:, 1], m.predict_proba(np.c_[xq, np.ones(len(xq))])[:, 1]
    out = []
    for a in (0, 1):
        m = LogisticRegression(max_iter=2000).fit(x[t == a], y[t == a])
        out.append(m.predict_proba(xq)[:, 1])
    return out[0], out[1]
