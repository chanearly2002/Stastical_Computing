"""Numerical check of Lemma 1 and Theorem 1 (Sec 3.2) on discrete synthetic problems.

x lives on K points of the real line, squared loss, Phi = identity (so R = X).
Quantities (noise variance s_t(x) = Var(Y_t|x)):
  l(x,t)  = (f(x,t) - m_t(x))^2 + s_t(x)
  eps_F, eps_CF, eps_F^{t=1}, eps_F^{t=0}, PEHE, and sig = sum_x p(x) (s_1 + s_0)
  (our reconstruction of the paper's "2 sigma_Y^2" term).
Two IPMs: (a) G = {|g|<=1}  -> sum|p1-p0|,  B = max l
          (b) G = 1-Lipschitz -> Wasserstein-1 on the line,  B = max_t Lip(l(.,t)).
"""
import sys
import numpy as np


def quantities(x, pX, e, m0, m1, f0, f1, s0, s1):
    p_joint1, p_joint0 = pX * e, pX * (1 - e)
    u = p_joint1.sum()
    p1, p0 = p_joint1 / u, p_joint0 / (1 - u)           # p(x|t=1), p(x|t=0)
    l1, l0 = (f1 - m1) ** 2 + s1, (f0 - m0) ** 2 + s0    # l(x,1), l(x,0)
    eps_F = (l1 * p_joint1 + l0 * p_joint0).sum()
    eps_CF = (l1 * p_joint0 + l0 * p_joint1).sum()        # measure p(x,1-t)
    eps1, eps0 = (l1 * p1).sum(), (l0 * p0).sum()
    pehe = (pX * ((f1 - f0) - (m1 - m0)) ** 2).sum()
    sig = (pX * (s1 + s0)).sum()
    # IPMs
    ipm_tv = np.abs(p1 - p0).sum()
    B_tv = max(l1.max(), l0.max())
    dx = np.diff(x)
    ipm_w = (np.abs(np.cumsum(p1 - p0)[:-1]) * dx).sum()
    B_w = max((np.abs(np.diff(l1)) / dx).max(), (np.abs(np.diff(l0)) / dx).max())
    return dict(u=u, eps_F=eps_F, eps_CF=eps_CF, eps1=eps1, eps0=eps0, pehe=pehe,
                sig=sig, ipm_tv=ipm_tv, B_tv=B_tv, ipm_w=ipm_w, B_w=B_w,
                sum_check=(l1 * pX).sum() + (l0 * pX).sum())


def random_problem(rng, K=6):
    x = np.sort(rng.uniform(0, 1, K))
    pX = rng.dirichlet(np.ones(K))
    e = rng.uniform(0.05, 0.95, K)
    m0, m1 = rng.normal(0, 1, K), rng.normal(0, 1, K)
    f0, f1 = m0 + rng.normal(0, .7, K), m1 + rng.normal(0, .7, K)
    s0, s1 = rng.uniform(0, .5, K), rng.uniform(0, .5, K)
    return x, pX, e, m0, m1, f0, f1, s0, s1


def check(q, tol=1e-12):
    ok = {}
    ok["eps_F = u*eps1 + (1-u)*eps0"] = abs(q["eps_F"] - (q["u"] * q["eps1"] + (1 - q["u"]) * q["eps0"])) < tol
    ok["eps_F + eps_CF = sum_t int l p(x)"] = abs(q["eps_F"] + q["eps_CF"] - q["sum_check"]) < tol
    ok["PEHE <= 2(eps_F+eps_CF-sig)"] = q["pehe"] <= 2 * (q["eps_F"] + q["eps_CF"] - q["sig"]) + tol
    mix = (1 - q["u"]) * q["eps1"] + q["u"] * q["eps0"]
    for name in ("tv", "w"):
        ok[f"Lemma1 [{name}]"] = q["eps_CF"] <= mix + q["B_" + name] * q["ipm_" + name] + tol
        rhs = 2 * (q["eps1"] + q["eps0"] + q["B_" + name] * q["ipm_" + name] - q["sig"])
        ok[f"Thm1 final [{name}]"] = q["pehe"] <= rhs + tol
    return ok


def toy():
    # Sec 3.1 toy: x in {R=0, P=1}; noise-free; f wrong at (P,1)
    x = np.array([0.0, 1.0]); pX = np.array([.4, .6]); e = np.array([.75, 1 / 6])
    m0, m1 = np.array([2., 1.]), np.array([5., 2.])
    f0, f1 = m0.copy(), np.array([5., 4.])
    z = np.zeros(2)
    return quantities(x, pX, e, m0, m1, f0, f1, z, z)


if __name__ == "__main__":
    q = toy()
    print("TOY: eps_F=%.3f eps_CF=%.3f eps1=%.3f eps0=%.3f PEHE=%.3f" % (q["eps_F"], q["eps_CF"], q["eps1"], q["eps0"], q["pehe"]))
    mix = (1 - q["u"]) * q["eps1"] + q["u"] * q["eps0"]
    print("     Lemma1: eps_CF=%.3f <= %.3f + B*W = %.3f + %.3f*%.4f = %.3f" % (q["eps_CF"], mix, mix, q["B_w"], q["ipm_w"], mix + q["B_w"] * q["ipm_w"]))
    print("     chain : PEHE %.3f <= 2(eF+eCF)=%.3f <= 2(e1+e0+B*W)=%.3f" % (q["pehe"], 2 * (q["eps_F"] + q["eps_CF"]), 2 * (q["eps1"] + q["eps0"] + q["B_w"] * q["ipm_w"])))
    print("     checks:", check(q))

    rng = np.random.default_rng(0)
    N = 100000
    fails = {}
    tight = {"tv": 0.0, "w": 0.0}
    for _ in range(N):
        q = quantities(*random_problem(rng))
        for k, v in check(q).items():
            if not v:
                fails[k] = fails.get(k, 0) + 1
        for name in ("tv", "w"):
            rhs = 2 * (q["eps1"] + q["eps0"] + q["B_" + name] * q["ipm_" + name] - q["sig"])
            tight[name] = max(tight[name], q["pehe"] / rhs)
    print(f"\n{N} random problems. violations: {fails if fails else 'none'}")
    print("largest PEHE / final-bound ratio (1.0 would be tight):", {k: round(v, 3) for k, v in tight.items()})
    sys.exit(1 if fails else 0)
