"""Aggregate results/ihdp_sweep.jsonl: Table-1-style comparison and Fig-2-style alpha sweep."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cfr.data import ihdp_rep, ihdp_split
from cfr import baselines
from cfr.metrics import sqrt_pehe, eps_ate

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/ihdp_sweep.jsonl"))]
by = collections.defaultdict(dict)                     # (ipm, alpha) -> rep -> row
for r in rows:
    by[(r["ipm"], r["alpha"])][r["rep"]] = r
grid = {i: sorted(a for (j, a) in by if j == i) for i in ("wass", "mmd")}
reps = sorted(set.intersection(*[set(by[k]) for k in by]))
print(f"{len(reps)} complete replications")


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), float(np.median(v))


# ---- baselines on the same splits ----
base = collections.defaultdict(lambda: collections.defaultdict(list))
ref = collections.defaultdict(list)
for r in reps:
    rep = ihdp_rep(r); sp = ihdp_split(rep, seed=r)
    x, t, y = rep["x"], rep["t"], rep["y"]; tau = rep["mu1"] - rep["mu0"]
    wi = np.concatenate([sp["tr"], sp["va"]]); te = sp["te"]
    for name, f in (("OLS1", baselines.ols1), ("OLS2", baselines.ols2), ("k-NN", baselines.knn), ("RF (T-learner)", baselines.forest)):
        f0, f1 = f(x[wi], t[wi], y[wi], x); th = f1 - f0
        base[name]["pehe_in"].append(sqrt_pehe(th[wi], tau[wi])); base[name]["ate_in"].append(eps_ate(th[wi], tau[wi]))
        base[name]["pehe_out"].append(sqrt_pehe(th[te], tau[te])); base[name]["ate_out"].append(eps_ate(th[te], tau[te]))
    ref["tau_sd_in"].append(float(tau[wi].std())); ref["tau_sd_out"].append(float(tau[te].std()))


def pick(ipm, how):
    """per-replication alpha chosen by the validation proxy ('proxy') or the validation factual loss ('loss')."""
    out = collections.defaultdict(list)
    key = {"proxy": "nn_proxy", "loss": "val_loss"}[how]
    for r in reps:
        cands = [by[(ipm, a)][r] for a in grid[ipm]]
        b = min(cands, key=lambda c: c[key])
        for k in ("pehe_in", "ate_in", "pehe_out", "ate_out"):
            out[k].append(b[k])
        out["alpha"].append(b["alpha"])
    return out


table = collections.OrderedDict()
for name in base:
    table[name] = base[name]
table["TARNet (alpha=0)"] = {k: [by[("none", 0.0)][r][k] for r in reps] for k in ("pehe_in", "ate_in", "pehe_out", "ate_out")}
table["CFR-MMD (alpha by val. proxy)"] = pick("mmd", "proxy")
table["CFR-Wass (alpha by val. proxy)"] = pick("wass", "proxy")
# best single alpha for the whole experiment, chosen by the mean validation proxy (no test information)
for ipm, lab in (("mmd", "CFR-MMD"), ("wass", "CFR-Wass")):
    a_best = min(grid[ipm], key=lambda a: np.mean([by[(ipm, a)][r]["nn_proxy"] for r in reps]))
    table[f"{lab} (one alpha={a_best} by mean val. proxy)"] = {k: [by[(ipm, a_best)][r][k] for r in reps] for k in ("pehe_in", "ate_in", "pehe_out", "ate_out")}

summary = {}
print("\n| method | sqrtPEHE in | eATE in | sqrtPEHE out | eATE out | median sqrtPEHE in/out |\n|---|---|---|---|---|---|")
for name, d in table.items():
    pi, ai, po, ao = (ms(d[k]) for k in ("pehe_in", "ate_in", "pehe_out", "ate_out"))
    summary[name] = {"pehe_in": pi, "ate_in": ai, "pehe_out": po, "ate_out": ao}
    print(f"| {name} | {pi[0]:.2f} ± {pi[1]:.2f} | {ai[0]:.2f} ± {ai[1]:.2f} | {po[0]:.2f} ± {po[1]:.2f} | {ao[0]:.2f} ± {ao[1]:.2f} | {pi[2]:.2f} / {po[2]:.2f} |")
print(f"\nreference: sd of the true tau over units (what a constant-effect predictor with the exact ATE would score): in {np.mean(ref['tau_sd_in']):.2f} (median {np.median(ref['tau_sd_in']):.2f}), out {np.mean(ref['tau_sd_out']):.2f} (median {np.median(ref['tau_sd_out']):.2f})")
for ipm in ("mmd", "wass"):
    a = pick(ipm, "proxy")["alpha"]
    print(ipm, "selected alpha counts:", dict(collections.Counter(a)))

# ---- Fig 2 style: out-of-sample error relative to alpha = 0 ----
rs = np.random.RandomState(0)
def rel(ipm, a, key="pehe_out"):
    return np.array([by[(ipm, a)][r][key] / by[("none", 0.0)][r][key] for r in reps])
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
curves = {}
for axi, key, title in ((ax[0], "pehe_in", "within-sample"), (ax[1], "pehe_out", "out-of-sample")):
    for ipm, col, lab in (("wass", "#B0214D", "CFR-Wass"), ("mmd", "#0B857E", "CFR-MMD")):
        xs = [0.0] + grid[ipm]; mean, lo, hi = [1.0], [1.0], [1.0]
        for a in grid[ipm]:
            v = rel(ipm, a, key); bs = [v[rs.randint(0, len(v), len(v))].mean() for _ in range(1000)]
            mean.append(v.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        curves[f"{ipm}_{key}"] = {"alpha": xs, "mean": mean, "lo": lo, "hi": hi}
        axi.plot(xs, mean, "o-", color=col, label=lab); axi.fill_between(xs, lo, hi, color=col, alpha=0.18)
    axi.set_xscale("symlog", linthresh=0.01); axi.axhline(1, color="k", lw=0.8, ls="--")
    axi.set_xlabel("IPM weight alpha (0 = TARNet)"); axi.set_title(f"IHDP {title}: sqrt(PEHE) relative to alpha=0\nmean of per-replication ratios, 95% bootstrap CI, {len(reps)} replications")
ax[0].set_ylabel("sqrt(PEHE) / sqrt(PEHE at alpha=0)"); ax[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures/09_ihdp_alpha_sweep.png"), dpi=130)
json.dump({"n_reps": len(reps), "table": summary, "curves": curves, "grid": grid}, open(os.path.join(ROOT, "results/ihdp_summary.json"), "w"), indent=1)
