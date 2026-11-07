"""Fig-2-style experiment on biased IHDP subsamples (q = 0, 0.5, 1): out-of-sample error of CFR-Wass relative to alpha = 0."""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/ihdp_imbalance.jsonl"))]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["q"], r["ipm"], r["alpha"])][r["rep"]] = r
qs = sorted({k[0] for k in by})
rs = np.random.RandomState(0); fig, ax = plt.subplots(figsize=(6.5, 4.4)); out = {}
cols = {0.0: "#1B6BB3", 0.5: "#2a8a3a", 1.0: "#B0214D"}
for q in qs:
    reps = sorted(set.intersection(*[set(by[k]) for k in by if k[0] == q]))
    al = sorted(a for (qq, i, a) in by if qq == q and i == "wass")
    xs, mean, lo, hi = [0.0], [1.0], [1.0], [1.0]
    for a in al:
        v = np.array([by[(q, "wass", a)][r]["pehe_out"] / by[(q, "none", 0.0)][r]["pehe_out"] for r in reps])
        bs = [v[rs.randint(0, len(v), len(v))].mean() for _ in range(1000)]
        xs.append(a); mean.append(v.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    ax.plot(xs, mean, "o-", color=cols.get(q, "k"), label=f"q = {q}"); ax.fill_between(xs, lo, hi, color=cols.get(q, "k"), alpha=0.15)
    out[str(q)] = {"n_reps": len(reps), "alpha": xs, "mean": mean, "lo": lo, "hi": hi}
    print(f"q={q}: {len(reps)} reps; relative out-of-sample sqrt(PEHE):", dict(zip(xs, np.round(mean, 3))))
ax.set_xscale("symlog", linthresh=0.01); ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_xlabel("IPM weight alpha (0 = TARNet)"); ax.set_ylabel("out-of-sample sqrt(PEHE) / value at alpha=0")
ax.set_title("IHDP biased subsamples, CFR-Wass (cf. paper Fig. 2)"); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures/11_ihdp_imbalance.png"), dpi=130)
json.dump(out, open(os.path.join(ROOT, "results/imbalance_summary.json"), "w"), indent=1)
