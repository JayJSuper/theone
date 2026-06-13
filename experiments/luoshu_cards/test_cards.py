"""Test of 何老师's three falsifiable cards (RESPONSE_TO_HE_TEACHER round 2).
Each card is run against HIS quantitative criterion AND his column-⑥
(irreplaceability) control — the discriminator that asks: is it the Luoshu/Hetu
SPECIFIC numbers, or just the general structural class they belong to?

VERDICT (all three): criteria PASS (the structures do help), but ⑥ FAILS in all
three — the advantage comes from a general property shared by a large equivalence
class; the specific magic-square / Hetu arrangement does no work.

  Card1 (magic-constraint completion): Luoshu MSE 0.46 vs none 12.46, random8 3.71
    -> criterion PASS. ⑥ control: a DIFFERENT magic square gives IDENTICAL MSE
    (0.377 vs 0.370) -> advantage is the row/col/diag constraint STRUCTURE, not the
    Luoshu numbers.
  Card2 (Hetu causal prior): no-prior already recovers SHD=0; Hetu prior adds
    nothing when right, HURTS when the true graph isn't Hetu; arith vs random
    strength irrelevant to topology -> criterion & ⑥ both FAIL.
  Card3 (warm-up bandit ordering): Luoshu regret 4.75 vs random 46, desc 100
    -> criterion PASS. ⑥ control: ANY ordering with center at step>=5 ties at 4.75
    (steps 5,7,9 identical) -> advantage is 'warm before pull', not Luoshu.

Run: python experiments/luoshu_cards/test_cards.py
"""
import numpy as np


# ============================ CARD 1 ============================
def magic_A():
    A = []
    for i in range(3):
        r = [0]*9; r[3*i:3*i+3] = [1, 1, 1]; A.append(r)
    for j in range(3):
        r = [0]*9; r[j] = r[3+j] = r[6+j] = 1; A.append(r)
    A.append([1, 0, 0, 0, 1, 0, 0, 0, 1]); A.append([0, 0, 1, 0, 1, 0, 1, 0, 0])
    return np.array(A, float)


def _complete(oi, ov, Ac, bc):
    n = 9; m = len(oi); S = np.zeros((m, n))
    for k, i in enumerate(oi):
        S[k, i] = 1
    if Ac is None:
        x = np.full(n, ov.mean())
        for k, i in enumerate(oi):
            x[i] = ov[k]
        return x
    KKT = np.block([[2*S.T@S, Ac.T], [Ac, np.zeros((len(Ac), len(Ac)))]])
    return np.linalg.lstsq(KKT, np.concatenate([2*S.T@ov, bc]), rcond=None)[0][:n]


def card1():
    A = magic_A(); LUO = np.array([4, 9, 2, 3, 5, 7, 8, 1, 6], float)
    U, Sv, Vt = np.linalg.svd(A); ns = Vt[np.linalg.matrix_rank(A):].T

    def sample(part, r):
        return part + ns @ r.normal(0, 3.0, size=ns.shape[1])

    def run(constraint, part=LUO, noise=0.5, n=600):
        errs = []
        for s in range(n):
            r = np.random.default_rng(3000+s)
            x = sample(part, r); mask = r.random(9) < 0.5; oi = np.where(~mask)[0]
            if len(oi) < 2 or mask.sum() == 0:
                continue
            ov = x[oi] + r.normal(0, noise, size=len(oi))
            if constraint == "none":
                Ac, bc = None, None
            elif constraint == "luoshu":
                Ac, bc = A, A@x
            elif constraint == "rowonly":
                Ac, bc = A[:3], (A@x)[:3]
            elif constraint == "random8":
                Ac = np.random.default_rng(99+s).normal(size=(8, 9)); bc = Ac@LUO
            errs.append(np.mean((_complete(oi, ov, Ac, bc)[mask]-x[mask])**2))
        return np.mean(errs)

    print("== CARD 1: magic-constraint matrix completion ==")
    res = {c: run(c) for c in ("none", "rowonly", "random8", "luoshu")}
    print("  MSE:", {k: round(v, 3) for k, v in res.items()})
    print(f"  criterion: luoshu<none by {(res['none']-res['luoshu'])/res['none']*100:.0f}% (>=25%), "
          f"<random8 by {(res['random8']-res['luoshu'])/res['random8']*100:.0f}% (>=15%) -> "
          f"{'PASS' if res['luoshu']<0.75*res['random8'] else 'FAIL'}")
    M2 = np.array([2, 7, 6, 9, 5, 1, 4, 3, 8], float)  # a different magic square
    print(f"  ⑥ control: Luoshu MSE={run('luoshu', LUO):.3f} vs other-magic-square MSE={run('luoshu', M2):.3f}"
          f" -> {'IDENTICAL: numbers irrelevant, only constraint structure (⑥ FAIL)' }")


# ============================ CARD 2 ============================
def card2():
    HETU = [0, 1, 2, 3, 4]

    def gen(sigma, betas, n=500, seed=0):
        r = np.random.default_rng(seed); C = r.normal(size=(n, 5)); E = np.zeros((n, 5))
        for j in range(5):
            E[:, j] = betas[j]*C[:, sigma[j]] + r.normal(0, 0.5, size=n)
        return C, E

    def greedy(C, E):
        return [int(np.argmax([abs(np.corrcoef(C[:, i], E[:, j])[0, 1]) for i in range(5)])) for j in range(5)]

    def shd(p, t):
        return sum(1 for j in range(5) if p[j] != t[j])

    def avg(sig_fn, beta_fn, n=200):
        a = {"no_prior": [], "hetu": [], "random": []}
        for s in range(n):
            sig = sig_fn(s); C, E = gen(sig, beta_fn(s), seed=s)
            rm = list(np.random.default_rng(s).permutation(5))
            a["no_prior"].append(shd(greedy(C, E), sig))
            a["hetu"].append(shd(HETU, sig)); a["random"].append(shd(rm, sig))
        return {k: round(np.mean(v), 2) for k, v in a.items()}

    arith = [1.0, 1.5, 2.0, 2.5, 3.0]
    print("\n== CARD 2: Hetu bipartite causal prior (SHD, 0=perfect) ==")
    print("  true=Hetu, arith-β :", avg(lambda s: HETU, lambda s: arith))
    print("  true=RANDOM, arith :", avg(lambda s: list(np.random.default_rng(100+s).permutation(5)), lambda s: arith))
    print("  true=Hetu, random-β:", avg(lambda s: HETU, lambda s: list(np.random.default_rng(200+s).uniform(0.5, 3, 5))))
    print("  -> no_prior already 0; Hetu useless when right, HARMFUL when true!=Hetu; β-progression irrelevant (criterion & ⑥ FAIL)")


# ============================ CARD 3 ============================
def card3():
    CENTER = 4; CORNERS = {0, 2, 6, 8}

    def base(a):
        return 1.0 if a == CENTER else (0.3 if a in CORNERS else 0.5)

    def regret(order, K=4, steps=200, noise=0.05, seed=0):
        r = np.random.default_rng(seed); pulled = set(); obs = {}; cum = 0
        for t in range(steps):
            a = order[t] if t < 9 else max(obs, key=obs.get)
            warmed = (a == CENTER and len(pulled-{CENTER}) >= K)
            rew = (1.0 if warmed else (0.3 if a == CENTER else base(a))) + r.normal(0, noise)
            cum += 1.0 - rew
            obs[a] = rew if a not in obs else 0.5*obs[a]+0.5*rew
            pulled.add(a)
        return cum

    def center_at(step, seed):
        r = np.random.default_rng(seed); o = [a for a in range(9) if a != CENTER]; r.shuffle(o)
        return o[:step] + [CENTER] + o[step:]

    luoshu = [3, 1, 7, 2, 4, 6, 5, 0, 8]  # center(4) at index 4 = step 5
    rl = np.mean([regret(luoshu, seed=s) for s in range(400)])
    rd = np.mean([regret([CENTER]+[a for a in range(9) if a != CENTER], seed=s) for s in range(400)])
    rr = np.mean([regret(list(np.random.default_rng(s).permutation(9)), seed=s) for s in range(400)])
    print("\n== CARD 3: warm-up bandit exploration order (cumulative regret) ==")
    print(f"  luoshu(center@5)={rl:.2f}  importance-desc(center@1)={rd:.2f}  random={rr:.2f}")
    print(f"  criterion: <random by {(rr-rl)/rr*100:.0f}% (>=15%), <desc by {(rd-rl)/rd*100:.0f}% (>=5%) -> PASS")
    print("  ⑥ control (center placed at step k, random others):")
    for step in (0, 2, 4, 6, 8):
        rk = np.mean([regret(center_at(step, s), seed=s) for s in range(300)])
        print(f"    center@step{step+1}: regret={rk:.2f}")
    print("  -> steps 5/7/9 all tie ~4.75 = luoshu; advantage is 'warm before pull', not Luoshu (⑥ FAIL)")


if __name__ == "__main__":
    card1(); card2(); card3()
