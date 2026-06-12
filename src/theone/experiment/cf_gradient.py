"""Counterfactual-gradient toy (T2-04, 站桩册②).

Dual-head MLP trained on a linear-Gaussian SCM family. Head 1 predicts the
observational slope (factual); head 2 predicts the interventional do-ATE
(counterfactual truth = beta_xy, available from the synthetic SCM). Loss =
MSE_factual + lambda * MSE_counterfactual. We test whether lambda>0 yields lower
held-out intervention error than the factual-only control (lambda=0) and the
pure-association baseline.

Pure numpy (no deep-learning dependency); all randomness seeded explicitly.
Criteria are frozen in experiments/cf_gradient_toy/prereg.md. This module does
NOT decide vital signs leniently - the thresholds live here verbatim.

Honest scope: lambda>0 directly supervises head 2, so beating the baseline on
intervention is not a surprise; the representation-transfer question is answered
by the T4 cross-family judgment gate, not here. Vital sign 1 is a sign-of-life
bar only.
"""
from __future__ import annotations
import math
import numpy as np
from ..bench.eg import SCMSpec, SyntheticSCM

FEATURE_DIM = 5  # [Cov(X,Y), Var(X), Cov(X,U), Cov(U,Y), Var(U)]


# ----------------------------- data ----------------------------------------
def _instance_features(spec: SCMSpec):
    """Observational summary stats (features) + factual/counterfactual labels."""
    scm = SyntheticSCM(spec)
    d = scm.sample()
    X, U, Y = d["X"], d["U"], d["Y"]
    cov_xy = float(np.cov(X, Y, bias=True)[0, 1])
    var_x = float(np.var(X))
    cov_xu = float(np.cov(X, U, bias=True)[0, 1])
    cov_uy = float(np.cov(U, Y, bias=True)[0, 1])
    var_u = float(np.var(U))
    phi = np.array([cov_xy, var_x, cov_xu, cov_uy, var_u], float)
    y_fac = cov_xy / var_x                     # observational slope
    y_cf = scm.true_int_ate()                  # do-ATE truth = beta_xy
    return phi, y_fac, y_cf


def build_dataset(n_instances: int, base_seed: int, n_samples: int = 500):
    """Sample an SCM family (params drawn from frozen ranges) -> (Phi, y_fac, y_cf)."""
    rng = np.random.default_rng(base_seed)
    Phi = np.empty((n_instances, FEATURE_DIM))
    y_fac = np.empty(n_instances)
    y_cf = np.empty(n_instances)
    for i in range(n_instances):
        p = float(rng.uniform(0.0, 0.8))
        b = math.sqrt(p)
        bxy = float(rng.uniform(0.0, 0.5))
        noise = float(rng.uniform(0.1, 0.5))
        spec = SCMSpec("linear_gaussian_3node", b, b, bxy, noise,
                       n_samples, base_seed + 1 + i)
        Phi[i], y_fac[i], y_cf[i] = _instance_features(spec)
    return Phi, y_fac, y_cf


# ----------------------------- model ---------------------------------------
class DualHeadMLP:
    """trunk 5->16->16 (tanh) + two linear heads (16->1). Adam; manual backprop."""

    def __init__(self, seed: int, hidden: int = 16, lr: float = 0.01):
        rng = np.random.default_rng(seed)
        def he(n_in, n_out):
            return rng.standard_normal((n_in, n_out)) * math.sqrt(2.0 / n_in)
        self.W1, self.b1 = he(FEATURE_DIM, hidden), np.zeros(hidden)
        self.W2, self.b2 = he(hidden, hidden), np.zeros(hidden)
        self.Wf, self.bf = he(hidden, 1), np.zeros(1)   # factual head
        self.Wc, self.bc = he(hidden, 1), np.zeros(1)   # counterfactual head
        self.lr = lr
        self._params = ["W1", "b1", "W2", "b2", "Wf", "bf", "Wc", "bc"]
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._t = 0

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1; a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.tanh(z2)
        pf = (a2 @ self.Wf + self.bf).ravel()
        pc = (a2 @ self.Wc + self.bc).ravel()
        return pf, pc, (X, z1, a1, z2, a2)

    def predict_cf(self, X):
        return self._forward(X)[1]

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for k in self._params:
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            setattr(self, k, getattr(self, k) - self.lr * mhat / (np.sqrt(vhat) + eps))

    def train(self, X, y_fac, y_cf, lam: float, epochs: int = 300):
        n = len(X)
        hist_f, hist_c = [], []
        for _ in range(epochs):
            pf, pc, (Xin, z1, a1, z2, a2) = self._forward(X)
            rf, rc = pf - y_fac, pc - y_cf
            hist_f.append(float(np.mean(rf ** 2)))
            hist_c.append(float(np.mean(rc ** 2)))
            # gradients of L = mean(rf^2) + lam*mean(rc^2)
            dpf = (2.0 / n) * rf            # dL/dpf
            dpc = (2.0 / n) * lam * rc      # dL/dpc
            gWf = a2.T @ dpf[:, None]; gbf = np.array([dpf.sum()])
            gWc = a2.T @ dpc[:, None]; gbc = np.array([dpc.sum()])
            da2 = dpf[:, None] @ self.Wf.T + dpc[:, None] @ self.Wc.T
            dz2 = da2 * (1 - a2 ** 2)
            gW2 = a1.T @ dz2; gb2 = dz2.sum(0)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - a1 ** 2)
            gW1 = Xin.T @ dz1; gb1 = dz1.sum(0)
            self._adam_step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
                             "Wf": gWf, "bf": gbf, "Wc": gWc, "bc": gbc})
        return hist_f, hist_c


# ------------------------- vital-sign judges (frozen) ----------------------
def vital_sign_1(hist_f, hist_c, drop: float = 0.01, run: int = 3) -> bool:
    """3 consecutive epochs where BOTH head losses drop >= `drop` relative."""
    def rel_drops(h):
        h = np.asarray(h, float)
        prev = np.maximum(h[:-1], 1e-12)
        return (h[:-1] - h[1:]) / prev          # relative drop per step
    df, dc = rel_drops(hist_f), rel_drops(hist_c)
    both = (df >= drop) & (dc >= drop)
    c = 0
    for ok in both:
        c = c + 1 if ok else 0
        if c >= run:
            return True
    return False


def _bca_lower(boot: np.ndarray, theta_hat: float, jack: np.ndarray,
               alpha: float = 0.05) -> float:
    from statistics import NormalDist
    nd = NormalDist()
    boot = np.asarray(boot, float)
    prop = np.clip(np.mean(boot < theta_hat), 1e-9, 1 - 1e-9)
    z0 = nd.inv_cdf(prop)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3); den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    z = nd.inv_cdf(alpha / 2)
    adj = nd.cdf(z0 + (z0 + z) / max(1e-12, (1 - a * (z0 + z))))
    return float(np.quantile(boot, np.clip(adj, 0.0, 1.0)))


def baseline_intervention_mse_ci(y_fac_holdout, y_cf_holdout, seed: int,
                                  B: int = 1000):
    """Pure-association baseline predicts do-ATE = observational slope.
    Returns (point MSE, BCa 95% lower bound) over held-out instances."""
    err2 = (np.asarray(y_fac_holdout) - np.asarray(y_cf_holdout)) ** 2
    theta = float(err2.mean())
    rng = np.random.default_rng(seed)
    n = len(err2)
    boot = np.array([err2[rng.integers(0, n, n)].mean() for _ in range(B)])
    jack = np.array([np.delete(err2, i).mean() for i in range(min(n, 200))])
    return theta, _bca_lower(boot, theta, jack)


# ------------------------------ experiment ---------------------------------
def run_experiment(lambdas=(0.0, 0.5, 1.0, 2.0), seeds=range(20),
                   n_train: int = 512, n_holdout: int = 256, epochs: int = 300,
                   train_seed: int = 300000, holdout_seed: int = 700000):
    Phi_tr, yf_tr, yc_tr = build_dataset(n_train, train_seed)
    Phi_ho, yf_ho, yc_ho = build_dataset(n_holdout, holdout_seed)
    mu, sd = Phi_tr.mean(0), Phi_tr.std(0) + 1e-9
    Xtr, Xho = (Phi_tr - mu) / sd, (Phi_ho - mu) / sd

    base_mse, base_lo = baseline_intervention_mse_ci(yf_ho, yc_ho, seed=12345)
    results = {"baseline_intervention_mse": base_mse,
               "baseline_bca95_lower": base_lo, "conditions": {}}
    for lam in lambdas:
        seed_rows = []
        for s in seeds:
            net = DualHeadMLP(seed=1000 + int(s))
            hf, hc = net.train(Xtr, yf_tr, yc_tr, lam=lam, epochs=epochs)
            cf_pred = net.predict_cf(Xho)
            int_mse = float(np.mean((cf_pred - yc_ho) ** 2))
            seed_rows.append({"seed": int(s), "vital1": vital_sign_1(hf, hc),
                              "final_loss_fac": hf[-1], "final_loss_cf": hc[-1],
                              "holdout_intervention_mse": int_mse})
        int_mses = np.array([r["holdout_intervention_mse"] for r in seed_rows])
        frac_v1 = float(np.mean([r["vital1"] for r in seed_rows]))
        med_int = float(np.median(int_mses))
        results["conditions"][f"lambda={lam}"] = {
            "vital1_fraction": frac_v1,
            "median_holdout_intervention_mse": med_int,
            "vital2_beats_baseline_lower": bool(med_int < base_lo),
            "seeds": seed_rows}
    return results
