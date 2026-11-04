"""Figure for Sec 3.2: (a) bound chain on the toy example, (b) PEHE vs bound on random problems,
(c) imbalance -> larger eps_CF / eps_F and larger IPM when errors concentrate where data are sparse."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_thm1 import quantities, random_problem, toy

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) toy chain
q = toy()
vals = [q["pehe"], 2 * (q["eps_F"] + q["eps_CF"]), 2 * (q["eps1"] + q["eps0"] + q["B_w"] * q["ipm_w"])]
labels = ["PEHE", "2(eps_F +\neps_CF)", "2(eps_F^1 + eps_F^0\n+ B*Wass)"]
bars = ax[0].bar(labels, vals, color=["#4c78a8", "#f58518", "#e45756"])
for b, v in zip(bars, vals):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center")
ax[0].set_ylabel("value"); ax[0].set_title("(a) Toy example: Theorem 1 chain (noise-free)")

# (b) random problems
rng = np.random.default_rng(1)
P, R = [], []
for _ in range(3000):
    qq = quantities(*random_problem(rng))
    P.append(qq["pehe"]); R.append(2 * (qq["eps1"] + qq["eps0"] + qq["B_w"] * qq["ipm_w"] - qq["sig"]))
ax[1].scatter(R, P, s=4, alpha=.4)
lim = max(max(R), max(P)); ax[1].plot([0, lim], [0, lim], "k--", lw=1, label="PEHE = bound")
ax[1].set_xscale("symlog", linthresh=1); ax[1].set_yscale("symlog", linthresh=1)
ax[1].set_xlabel("RHS of Eq.(2) with Wasserstein IPM"); ax[1].set_ylabel("PEHE")
ax[1].set_title("(b) 3000 random problems: PEHE <= bound"); ax[1].legend()

# (c) imbalance experiment
def sim(strength, rng, n=200, K=8, trials=1500):
    cf = f = ipm = pehe = 0.0
    for _ in range(trials):
        x = np.linspace(0, 1, K); pX = np.ones(K) / K
        z = (x - x.mean()) / x.std(); e = 1 / (1 + np.exp(-strength * z))
        m0, m1 = np.zeros(K), np.zeros(K)
        pj1, pj0 = pX * e, pX * (1 - e)
        f1 = rng.normal(0, 1, K) / np.sqrt(n * pj1)   # error variance ~ 1/(#samples in cell)
        f0 = rng.normal(0, 1, K) / np.sqrt(n * pj0)
        qq = quantities(x, pX, e, m0, m1, f0, f1, np.zeros(K), np.zeros(K))
        cf += qq["eps_CF"]; f += qq["eps_F"]; ipm += qq["ipm_w"]; pehe += qq["pehe"]
    return cf / f, ipm / trials, pehe / trials

S = np.linspace(0, 5, 11); out = np.array([sim(s, rng) for s in S])
ax[2].plot(S, out[:, 0], "o-", color="#e45756"); ax[2].set_xlabel("imbalance strength s  (s=0: randomized)")
ax[2].set_yscale("log"); ax[2].set_ylabel("mean eps_CF / mean eps_F (log)", color="#e45756")
a2 = ax[2].twinx(); a2.plot(S, out[:, 1], "s--", color="#4c78a8"); a2.set_ylabel("mean Wasserstein(p^{t=1}, p^{t=0})", color="#4c78a8")
ax[2].set_title("(c) More imbalance -> larger eps_CF / eps_F and IPM")
plt.tight_layout(); plt.savefig("figures/03_thm1_chain.png", dpi=130)
print("ratio eps_CF/eps_F and mean W1 by s:"); [print(f"s={s:.1f}  ratio={r:.2f}  W1={w:.3f}  PEHE={p:.3f}") for s, (r, w, p) in zip(S, out)]
