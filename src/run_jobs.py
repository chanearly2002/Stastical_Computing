"""Jobs experiment (paper Sec 5.2): binary outcome (1 = employed in the data files), log-loss, 10 splits.
Evaluates eps_ATT and policy risk (lambda = 0) on the randomized part, within-sample (train+val) and out-of-sample.
Usage: python src/run_jobs.py --splits 0 10 --out results/jobs.jsonl"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
RATES = [i / 20 for i in range(21)]


def _init():
    import torch
    torch.set_num_threads(1)


def _std(tr, others):
    import numpy as np
    cont = [j for j in range(tr["x"].shape[1]) if len(np.unique(tr["x"][:, j])) > 2]
    m, s = tr["x"][:, cont].mean(0), tr["x"][:, cont].std(0) + 1e-8
    for d in [tr] + others:
        d["x"] = d["x"].copy(); d["x"][:, cont] = (d["x"][:, cont] - m) / s


def task(args):
    split, method, alpha, iters = args
    import numpy as np
    from cfr.data import jobs_split
    from cfr.train import fit, predict
    from cfr import baselines
    from cfr.metrics import att_error, policy_risk, policy_risk_curve
    tr, va, te = jobs_split(split, seed=split)
    _std(tr, [va, te])
    wi = {k: np.concatenate([tr[k], va[k]]) for k in tr}
    t0 = time.time()
    if method in ("tarnet", "mmd", "wass"):
        net, info = fit(tr["x"], tr["t"], tr["y"], va["x"], va["t"], va["y"], ipm="none" if method == "tarnet" else method,
                        alpha=0.0 if method == "tarnet" else alpha, loss="log", iters=iters, seed=split,
                        d_rep=200, d_head=200)
        pred = lambda x: predict(net, x, info, loss="log")
    elif method == "lr1":
        pred = lambda x: baselines.logreg(wi["x"], wi["t"], wi["y"], x)
    elif method == "lr2":
        pred = lambda x: baselines.logreg(wi["x"], wi["t"], wi["y"], x, separate=True)
    elif method == "rf":
        pred = lambda x: baselines.forest(wi["x"], wi["t"], wi["y"], x, seed=split, classify=True)
    else:
        raise ValueError(method)
    out = {"split": split, "method": method, "alpha": alpha}
    for name, d in (("in", wi), ("out", te)):
        f0, f1 = pred(d["x"])
        out["att_" + name] = att_error(f1, f0, d["t"], d["y"], d["e"])
        out["rpol_" + name] = policy_risk(f1, f0, d["t"], d["y"], d["e"], 0.0)
        if name == "out":
            out["curve_out"] = policy_risk_curve(f1, f0, d["t"], d["y"], d["e"], RATES).tolist()
    # validation policy risk (randomized validation units, model fit on train only) for alpha selection
    if method in ("tarnet", "mmd", "wass"):
        f0, f1 = predict(net, va["x"], info, loss="log")
        out["rpol_val"] = policy_risk(f1, f0, va["t"], va["y"], va["e"], 0.0)
        out["val_loss"] = info["val_loss"]
    out["secs"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs=2, default=[0, 10])
    ap.add_argument("--alphas_wass", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1])
    ap.add_argument("--alphas_mmd", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", default="results/jobs.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["split"], r["method"], r["alpha"]))
    jobs = []
    for s in range(*a.splits):
        for m, al in [("lr1", 0.0), ("lr2", 0.0), ("rf", 0.0), ("tarnet", 0.0)] + [("wass", x) for x in a.alphas_wass] + [("mmd", x) for x in a.alphas_mmd]:
            if (s, m, al) not in done:
                jobs.append((s, m, al, a.iters))
    print(len(jobs), "jobs", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex, open(a.out, "a") as f:
        for k, fu in enumerate(as_completed([ex.submit(task, j) for j in jobs]), 1):
            r = fu.result(); f.write(json.dumps(r) + "\n"); f.flush()
            if k % 5 == 0:
                print(f"{k}/{len(jobs)} split {r['split']} {r['method']} a={r['alpha']} rpol_out={r['rpol_out']:.3f} att_out={r['att_out']:.3f} ({r['secs']}s)", flush=True)
