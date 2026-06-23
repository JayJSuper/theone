"""Complete-form loop on CONTINUOUS data: perceive -> native-continuous -> product.

The full chain end-to-end on continuous outcomes (real products are continuous):
  noisy continuous SEQUENCES (the confounder observed only as streams)
    -> SSMPerception.perceive_features  (O(N) SSM -> continuous covariates)
    -> NativeVerifiableEngine.estimate_continuous  (TARNet ATE + replay + 3-zone)
    -> product credential (ask_data_causal_continuous)
Confounder biases the naive contrast; perceiving it from streams and adjusting recovers the
true continuous ATE. Longer sequences -> better perception -> tighter ATE.

Run:  .venv/bin/python experiments/native_perception_continuous/run.py
"""
from __future__ import annotations
import numpy as np

from theone.native.perception import SSMPerception
from theone.app import TheOneApp


def make(n, T, ate, seed):
    """Continuous SCM where a latent confounder is observed only as noisy streams.
    confounder c -> treatment t and outcome y; we never see c, only T-length noisy streams."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n).astype(np.float32)                       # latent confounder
    base = np.sin(np.linspace(0, 3, T))[None, :] * c[:, None]       # stream carries c
    streams = (base + rng.normal(scale=1.0, size=(n, T))).astype(np.float32)
    t = (rng.random(n) < 1 / (1 + np.exp(-1.2 * c))).astype(np.float32)   # c -> t
    y = (2.0 + 1.5 * c + ate * t + rng.normal(scale=0.5, size=n)).astype(np.float32)  # c,t -> y
    return streams, t, y


def main():
    print("=== complete-form loop · CONTINUOUS · perceive -> native -> product ===\n")
    n, ate = 1600, 3.0
    app = TheOneApp(domain=None, llm=None, memory_path=":memory:")

    # naive (confounded) contrast — what you get WITHOUT perceiving the confounder
    streams64, t, y = make(n, 64, ate, 0)
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    print(f"true ATE = {ate:.2f}")
    print(f"naive confounded contrast (no perception) = {naive:.3f}  (bias {naive - ate:+.3f})\n")

    print(f"{'T (stream len)':>14} {'perceived ATE':>14} {'|err|':>7} {'zone':>22} {'replay':>7}")
    last = None
    for T in (8, 24, 64):
        streams, t, y = make(n, T, ate, 0)
        sp = SSMPerception(hidden_dim=24, seed=0)
        X = sp.perceive_features(streams, k=4)              # perceive confounder as covariates
        r = app.ask_data_causal_continuous(X, t, y, covariate_sufficient=True)
        ate_hat = float(r["credential"]["claim"].split("=")[1])
        err = abs(ate_hat - ate)
        print(f"{T:>14} {ate_hat:>14.3f} {err:>7.3f} {r['zone']:>22} {str(r['replay_ok']):>7}")
        last = (err, r)
    app.close()

    err, r = last
    beats_naive = err < abs(naive - ate)
    print(f"\ncomplete-form gate (longest sequence T=64):")
    print(f"  perceived+adjusted beats naive confounded contrast . {'PASS' if beats_naive else 'FAIL'}")
    print(f"  replay-verified through the product ................ {'PASS' if r['replay_ok'] else 'FAIL'}")
    gate = beats_naive and r["replay_ok"]
    print(f"\n  >>> {'PASS — perceive->native-continuous->product loop closes on continuous data' if gate else 'CHECK'}")
    print("\nMeaning: the WHOLE complete-form chain now runs on continuous outcomes — a latent")
    print("confounder seen only as noisy streams is perceived (O(N) SSM) into continuous")
    print("covariates, adjusted on by the native engine, and delivered as a replay-verified")
    print("product credential. Honest: linear SSM summary + a feasibility bridge, not a tuned")
    print("perceptual model; residual bias remains and shrinks as sequences lengthen.")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
