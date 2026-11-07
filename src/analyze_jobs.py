"""Aggregate results/jobs.jsonl: Table-1-style Jobs columns and a Fig-3-style policy risk curve."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/jobs.jsonl"))]
RATES = [i / 20 for i in range(21)]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["method"], r["alpha"])][r["split"]] = r
splits = sorted(set.intersection(*[set(v) for v in by.values()]))
print(len(splits), "complete splits")
alphas = {m: sorted(a for (mm, a) in by if mm == m) for m in ("mmd", "wass")}


def ms(v):
    v = np.asarray(v, float); return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def select(m):
    """per-split alpha chosen by validation policy risk (randomized validation units); ties -> smaller alpha"""
    out = {}
    for s in splits:
        cands = sorted(((by[(m, a)][s]["rpol_val"], a) for a in alphas[m]))
        out[s] = by[(m, cands[0][1])][s]
    return out


pick = {"LR1": {s: by[("lr1", 0.0)][s] for s in splits}, "LR2": {s: by[("lr2", 0.0)][s] for s in splits},
        "RF (T-learner)": {s: by[("rf", 0.0)][s] for s in splits}, "TARNet": {s: by[("tarnet", 0.0)][s] for s in splits},
        "CFR-MMD (alpha by val. policy risk)": select("mmd"), "CFR-Wass (alpha by val. policy risk)": select("wass")}
summary = {}
print("\n| method | R_Pol in | eps_ATT in | R_Pol out | eps_ATT out |\n|---|---|---|---|---|")
for name, d in pick.items():
    cells = [ms([d[s][k] for s in splits]) for k in ("rpol_in", "att_in", "rpol_out", "att_out")]
    summary[name] = dict(zip(("rpol_in", "att_in", "rpol_out", "att_out"), cells))
    print(f"| {name} | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in cells) + " |")
for m in ("mmd", "wass"):
    print(m, "alpha chosen:", dict(collections.Counter(pick[f"CFR-{'MMD' if m=='mmd' else 'Wass'} (alpha by val. policy risk)"][s]["alpha"] for s in splits)))
print("fixed-alpha rows (out-of-sample R_Pol / eps_ATT):")
for (m, a), d in sorted(by.items()):
    if m in ("mmd", "wass"):
        print(f"  {m} alpha={a}: R_Pol out {np.mean([d[s]['rpol_out'] for s in splits]):.3f}  eps_ATT out {np.mean([d[s]['att_out'] for s in splits]):.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.4))
for name, col, lab in (("RF (T-learner)", "#444", "RF (T-learner)"), ("LR2", "#7a7", "LR2"), ("TARNet", "#C2500F", "TARNet"), ("CFR-MMD (alpha by val. policy risk)", "#0B857E", "CFR-MMD")):
    c = np.array([pick[name][s]["curve_out"] for s in splits]); m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(splits))
    ax.plot(RATES, m, label=lab, color=col); ax.fill_between(RATES, m - se, m + se, color=col, alpha=0.15)
c = np.array([pick["TARNet"][s]["curve_out"] for s in splits]).mean(0)
ax.plot([0, 1], [c[0], c[-1]], "k--", label="random policy")
ax.set_xlabel("treatment inclusion rate"); ax.set_ylabel("out-of-sample policy risk"); ax.set_title(f"Jobs: policy risk vs inclusion rate ({len(splits)} splits, mean ± SE)")
ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures/10_jobs_policy_curve.png"), dpi=130)
json.dump({"n_splits": len(splits), "table": summary, "rates": RATES,
           "curves": {n: np.array([pick[n][s]["curve_out"] for s in splits]).mean(0).tolist() for n in pick}},
          open(os.path.join(ROOT, "results/jobs_summary.json"), "w"), indent=1)
