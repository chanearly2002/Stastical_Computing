"""Steps 1-3 of the reproduction plan: a synthetic confounded problem with known tau(x).
x in R^10; treatment depends strongly on (x0, x1) => imbalance between groups; outcomes are nonlinear.
For each alpha we train CFR (Wasserstein or MMD) and report out-of-sample sqrt(PEHE) on fresh data.
Usage: python src/run_synthetic.py --seeds 0 20 --out results/synthetic.jsonl"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))


def _init():
    import torch
    torch.set_num_threads(1)


def make(n, seed, strength=2.0, d=10):
    import numpy as np
    rs = np.random.RandomState(seed)
    x = rs.randn(n, d)
    logit = strength * (x[:, 0] + 0.5 * x[:, 1])
    t = (rs.rand(n) < 1 / (1 + np.exp(-logit))).astype(float)
    f0 = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.3 * x[:, 2]
    tau = 1.0 + 0.8 * np.tanh(x[:, 2]) + 0.5 * x[:, 3] * (x[:, 4] > 0)
    y0 = f0 + 0.5 * rs.randn(n)
    y1 = f0 + tau + 0.5 * rs.randn(n)
    return x, t, np.where(t == 1, y1, y0), tau


def task(args):
    seed, ipm, alpha, iters = args
    import numpy as np
    from cfr.train import fit, predict
    from cfr.metrics import sqrt_pehe
    from cfr.ipm import exact_wass_1d
    x, t, y, tau = make(500, seed)
    xt, _, _, taut = make(2000, 10_000 + seed)               # fresh sample from the same population
    idx = np.random.RandomState(seed).permutation(500); tr, va = idx[:350], idx[350:]
    t0 = time.time()
    net, info = fit(x[tr], t[tr], y[tr], x[va], t[va], y[va], ipm=ipm, alpha=alpha, iters=iters, seed=seed)
    f0, f1 = predict(net, xt, info)
    # imbalance of the learned representation on the test inputs (1-D projection on the first PC, for reporting only)
    import torch
    with torch.no_grad():
        r = net.representation(torch.tensor(x, dtype=torch.float32))
    mt, mc = r[torch.tensor(t == 1)].mean(0), r[torch.tensor(t == 0)].mean(0)
    return {"seed": seed, "ipm": ipm, "alpha": alpha, "pehe_out": sqrt_pehe(f1 - f0, taut),
            "rep_mean_gap": float(torch.norm(mt - mc)), "frac_treated": float(t.mean()), "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default="results/synthetic.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["seed"], r["ipm"], r["alpha"]))
    jobs = [(s, i, x, a.iters) for s in range(*a.seeds) for i, x in [("none", 0.0)] + [(i, x) for i in ("wass", "mmd") for x in a.alphas] if (s, i, x) not in done]
    print(len(jobs), "jobs", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex, open(a.out, "a") as f:
        for k, fu in enumerate(as_completed([ex.submit(task, j) for j in jobs]), 1):
            r = fu.result(); f.write(json.dumps(r) + "\n"); f.flush()
            if k % 10 == 0:
                print(f"{k}/{len(jobs)} seed {r['seed']} {r['ipm']} a={r['alpha']} pehe_out={r['pehe_out']:.3f}", flush=True)
