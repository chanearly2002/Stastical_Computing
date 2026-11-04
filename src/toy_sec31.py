"""Sec 3.1 toy example (hand-made, noise-free): x in {R, P}, observational assignment.
Checks eps_F, eps_CF, PEHE and a Wasserstein distance (two-point space, d(R,P)=1)."""
pX = {"R": 0.4, "P": 0.6}
pt1 = {"R": 0.75, "P": 1 / 6}                      # p(t=1|x)
m = {("R", 0): 2, ("R", 1): 5, ("P", 0): 1, ("P", 1): 2}   # true m_t(x)
f = {("R", 0): 2, ("R", 1): 5, ("P", 0): 1, ("P", 1): 4}   # hypothesis, wrong at (P,1)

joint = {(x, 1): pX[x] * pt1[x] for x in pX} | {(x, 0): pX[x] * (1 - pt1[x]) for x in pX}
cf = {(x, t): joint[(x, 1 - t)] for (x, t) in joint}       # p(x, 1-t)
loss = {k: (f[k] - m[k]) ** 2 for k in m}                  # noise-free: l(x,t)

u = sum(joint[(x, 1)] for x in pX)
eps_F = sum(loss[k] * joint[k] for k in joint)
eps_CF = sum(loss[k] * cf[k] for k in joint)
p1 = {x: joint[(x, 1)] / u for x in pX}                    # p(x|t=1)
p0 = {x: joint[(x, 0)] / (1 - u) for x in pX}              # p(x|t=0)
eF1 = sum(loss[(x, 1)] * p1[x] for x in pX)
eF0 = sum(loss[(x, 0)] * p0[x] for x in pX)
tau = {x: m[(x, 1)] - m[(x, 0)] for x in pX}
tau_hat = {x: f[(x, 1)] - f[(x, 0)] for x in pX}
pehe = sum(pX[x] * (tau_hat[x] - tau[x]) ** 2 for x in pX)
w1 = abs(p1["R"] - p0["R"])                                # W1 on 2-point space, d=1

print(f"u={u:.3f} p(x|t=1)={p1} p(x|t=0)={p0}")
print(f"eps_F={eps_F:.3f} (check u*eF1+(1-u)*eF0={u*eF1+(1-u)*eF0:.3f}) eF1={eF1:.3f} eF0={eF0:.3f}")
print(f"eps_CF={eps_CF:.3f}  PEHE={pehe:.3f}  2*(eps_F+eps_CF)={2*(eps_F+eps_CF):.3f}")
print(f"CF propensity p(t=1 in CF world | P) = {1-pt1['P']:.4f};  Wass(p1,p0)={w1:.4f}")
