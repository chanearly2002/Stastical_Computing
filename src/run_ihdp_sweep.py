"""IHDP experiment: for each outcome realization, train TARNet (alpha=0) and CFR-MMD / CFR-Wass for a grid of alpha.
Usage: python src/run_ihdp_sweep.py --reps 0 60 --out results/ihdp_sweep.jsonl
Resumable: finished (rep, ipm, alpha, q) keys in --out are skipped. Optional --q builds the biased subsamples of
paper Sec 5.1 (q in [0,1]; 400 units kept)."""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def _init():
    import torch
    torch.set_num_threads(1)


def task(args):
    rep_i, ipm, alpha, q, iters, seed = args
    import numpy as np
    from cfr.data import ihdp_rep, ihdp_split, ihdp_biased, split_indices
    from cfr.experiment import run_ihdp
    rep = ihdp_rep(rep_i)
    if q is None:
        sp = ihdp_split(rep, seed=rep_i)
    else:
        keep = ihdp_biased(rep, q, seed=rep_i)
        rep = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in rep.items()}
        s = split_indices(len(keep), seed=rep_i)
        sp = {"tr": s["tr"], "va": s["va"], "te": s["te"]}
        rep["n_train"] = len(keep)
    t0 = time.time()
    r = run_ihdp(rep, sp, ipm=ipm, alpha=alpha, seed=seed, iters=iters)
    r.update(rep=rep_i, ipm=ipm, alpha=alpha, q=q, seed=seed, secs=round(time.time() - t0, 1))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs=2, default=[0, 10])
    ap.add_argument("--alphas_wass", type=float, nargs="*", default=[0.01, 0.03, 0.1, 0.3, 1])
    ap.add_argument("--alphas_mmd", type=float, nargs="*", default=[0.1, 0.3, 1, 3, 10])
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", default="results/ihdp_sweep.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["rep"], r["ipm"], r["alpha"], r["q"]))
    jobs = []
    for rep in range(a.reps[0], a.reps[1]):
        for ipm, al in [("none", 0.0)] + [("wass", x) for x in a.alphas_wass] + [("mmd", x) for x in a.alphas_mmd]:
            if (rep, ipm, al, a.q) not in done:
                jobs.append((rep, ipm, al, a.q, a.iters, 0))
    print(f"{len(jobs)} jobs", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex, open(a.out, "a") as f:
        futs = [ex.submit(task, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); f.write(json.dumps(r) + "\n"); f.flush()
            if k % 10 == 0:
                print(f"{k}/{len(jobs)} last: rep {r['rep']} {r['ipm']} a={r['alpha']} pehe_out={r['pehe_out']:.3f} ({r['secs']}s)", flush=True)
