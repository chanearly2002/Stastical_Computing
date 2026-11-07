"""Aggregate results/synthetic.jsonl: out-of-sample sqrt(PEHE) and representation gap vs alpha."""
import json, os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/synthetic.jsonl"))]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["ipm"], r["alpha"])][r["seed"]] = r
seeds = sorted(set.intersection(*[set(v) for v in by.values()]))
print(len(seeds), "seeds; frac treated mean", np.mean([by[("none", 0.0)][s]["frac_treated"] for s in seeds]).round(3))
base = np.array([by[("none", 0.0)][s]["pehe_out"] for s in seeds])
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); out = {"base": [float(base.mean()), float(base.std(ddof=1) / np.sqrt(len(base)))]}
for ipm, col, lab in (("wass", "#B0214D", "CFR-Wass"), ("mmd", "#0B857E", "CFR-MMD")):
    al = sorted(a for (i, a) in by if i == ipm)
    rel = np.array([[by[(ipm, a)][s]["pehe_out"] / by[("none", 0.0)][s]["pehe_out"] for s in seeds] for a in al])
    gap = np.array([[by[(ipm, a)][s]["rep_mean_gap"] for s in seeds] for a in al]); g0 = np.mean([by[("none", 0.0)][s]["rep_mean_gap"] for s in seeds])
    m, se = rel.mean(1), rel.std(1, ddof=1) / np.sqrt(len(seeds))
    ax[0].errorbar(al, m, se, fmt="o-", color=col, label=lab, capsize=3)
    ax[1].plot([0.003] + al, [g0] + list(gap.mean(1)), "o-", color=col, label=lab)
    out[ipm] = {"alpha": al, "rel_mean": m.tolist(), "rel_se": se.tolist(), "gap": gap.mean(1).tolist(), "gap0": float(g0)}
    print(ipm, "rel out-of-sample sqrt(PEHE):", dict(zip(al, m.round(3))), "\n   rep. gap:", dict(zip(al, gap.mean(1).round(3))), "gap at alpha=0:", round(float(g0), 3))
ax[0].axhline(1, color="k", ls="--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_xlabel("alpha"); ax[0].set_ylabel("sqrt(PEHE) relative to alpha=0 (out-of-sample)")
ax[0].set_title(f"Synthetic confounded data, {len(seeds)} seeds, mean ± SE"); ax[0].legend()
ax[1].set_xscale("log"); ax[1].set_xlabel("alpha (leftmost point: alpha=0)"); ax[1].set_ylabel("|| mean Phi(treated) - mean Phi(control) ||"); ax[1].set_title("Representation imbalance shrinks with alpha"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures/08_synthetic_alpha.png"), dpi=130)
json.dump({"n_seeds": len(seeds), **out}, open(os.path.join(ROOT, "results/synthetic_summary.json"), "w"), indent=1)
