"""Fusion Phase A · runnable proof of the credential spine across all 6 layers.

This is the first executable artifact of the fusion (docs/FUSION_ARCHITECTURE.md):
it wires one CredentialedLayer per L0..L5, each enforcing TWO orthogonal gates —
  • admissibility: a blueprint threshold (energy drift, reconstruction MSE, VFE
    convergence, ...) → ABSTAIN if violated;
  • recomputability: a Credential whose value is independently reproduced → the
    spine auto-downgrades any non-reproducing ANSWER to ABSTAIN.
Together they are os_loop_constrained's two gates generalized to a 6-layer bus.

Three scenarios prove the soul of the design:
  A. healthy   → system ANSWER with 6 stacked, independently-recomputed credentials.
  B. L0 fault  → system ABSTAINS at the physics gate (energy drift > 1e-3); no
                 confident-wrong output is produced downstream.
  C. lying L2  → a layer emits a credential whose claim disagrees with its own
                 recomputation; the spine catches it and ABSTAINS — trust is from
                 recomputation, never confidence (the LLM failure mode, structurally barred).

Run:  .venv/bin/python experiments/fusion_spine/run.py
"""
from __future__ import annotations
import hashlib
import numpy as np

from theone.core.spine import Decision, Credential, LayerVerdict, CredentialedLayer, Spine


# ----- one minimal-but-real gate per layer (admissibility + recomputability) --------
class L0Physics(CredentialedLayer):
    name, layer_index = "L0_physics", 0
    TOL = 1e-3

    def process(self, ctx):
        h0, h1 = ctx["energies"]                      # before/after a state evolution
        drift = abs(h1 - h0)
        if drift > self.TOL:                          # admissibility gate
            return LayerVerdict.abstain(self.name, f"energy drift {drift:.2e} > {self.TOL}")
        cred = Credential(self.name, "symplectic evolution conserves energy",
                          value=drift, regime="valid where latent state has Hamiltonian structure",
                          recompute=lambda: abs(ctx["energies"][1] - ctx["energies"][0]),
                          tolerance=1e-12, evidence={"threshold": self.TOL})
        return LayerVerdict.answer(self.name, cred, value=ctx)


class L1Perception(CredentialedLayer):
    name, layer_index = "L1_perception", 1
    TOL = 1e-3

    def process(self, ctx):
        sig, recon = np.asarray(ctx["signal"]), np.asarray(ctx["recon"])
        mse = float(np.mean((sig - recon) ** 2))
        if mse > self.TOL:
            return LayerVerdict.abstain(self.name, f"reconstruction MSE {mse:.2e} > {self.TOL}")
        cred = Credential(self.name, "SSM encoding reconstructs the signal",
                          value=mse, regime="linear-decoder reconstruction",
                          recompute=lambda: float(np.mean((np.asarray(ctx["signal"])
                                                           - np.asarray(ctx["recon"])) ** 2)),
                          tolerance=1e-12)
        return LayerVerdict.answer(self.name, cred, value=ctx)


class L2Causal(CredentialedLayer):
    name, layer_index = "L2_causal", 2

    def process(self, ctx):
        do_a, do_b = ctx["do_method_a"], ctx["do_method_b"]   # engine vs IPRG recompute
        # the layer CLAIMS do_a; recomputation re-derives via the second method.
        claimed = ctx.get("claimed_do", do_a)
        cred = Credential(self.name, "P(Y|do(X)) under the assumed structure",
                          value=claimed,
                          regime="computation-exact, structure-assumed (NOTE-004)",
                          recompute=lambda: ctx["do_method_b"], tolerance=1e-6,
                          evidence={"method_a": do_a, "method_b": do_b})
        return LayerVerdict.answer(self.name, cred, value=ctx)


class L3Decision(CredentialedLayer):
    name, layer_index = "L3_decision", 3
    F_THRESHOLD = 0.01

    def process(self, ctx):
        trace = np.asarray(ctx["vfe_trace"], dtype=float)
        monotone = bool(np.all(np.diff(trace) <= 1e-12))
        if not monotone:
            return LayerVerdict.abstain(self.name, "variational free energy not monotone-decreasing")
        if trace[-1] > self.F_THRESHOLD:
            return LayerVerdict.abstain(self.name, f"VFE {trace[-1]:.3f} did not reach {self.F_THRESHOLD}")
        cred = Credential(self.name, "active-inference loop minimized free energy",
                          value=float(trace[-1]), regime="convergence on the given objective",
                          recompute=lambda: float(np.asarray(ctx["vfe_trace"])[-1]),
                          tolerance=1e-12)
        return LayerVerdict.answer(self.name, cred, value=ctx)


class L4Memory(CredentialedLayer):
    name, layer_index = "L4_memory", 4

    @staticmethod
    def _sig(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def process(self, ctx):
        content, claimed_sig = ctx["mem_content"], ctx["claimed_sig"]
        if self._sig(content) != claimed_sig:
            return LayerVerdict.abstain(self.name, "causal signature does not match stored content")
        cred = Credential(self.name, "memory retrieved by exact causal signature",
                          value=claimed_sig, regime="signature-indexed retrieval",
                          recompute=lambda: self._sig(ctx["mem_content"]), tolerance=0.0)
        return LayerVerdict.answer(self.name, cred, value=ctx)


class L5Execution(CredentialedLayer):
    name, layer_index = "L5_execution", 5
    BLACKLIST = ("rm -rf /", "chmod 777 /", ":(){ :|:& };:")

    def process(self, ctx):
        action = ctx["action"]
        cmd = action.get("cmd", "")
        if any(b in cmd for b in self.BLACKLIST):
            return LayerVerdict.abstain(self.name, f"reference monitor blocked dangerous command: {cmd!r}")
        if action.get("path") and not str(action["path"]).startswith("/sandbox/"):
            return LayerVerdict.abstain(self.name, f"path escapes sandbox: {action['path']}")
        digest = hashlib.sha256(repr(sorted(action.items())).encode()).hexdigest()[:16]
        cred = Credential(self.name, "action admissible + dry-run digest issued",
                          value=digest, regime="sandboxed dry-run",
                          recompute=lambda: hashlib.sha256(
                              repr(sorted(ctx["action"].items())).encode()).hexdigest()[:16],
                          tolerance=0.0)
        return LayerVerdict.answer(self.name, cred, value=ctx)


def healthy_ctx():
    t = np.linspace(0, 1, 200)
    sig = np.sin(2 * np.pi * t)
    return {
        "energies": (1.000000, 1.000004),                  # drift 4e-6 < 1e-3
        "signal": sig, "recon": sig + 1e-3 * np.sin(50 * t),  # tiny recon error
        "do_method_a": 0.726799, "do_method_b": 0.726799,  # engine == IPRG
        "vfe_trace": [0.5, 0.3, 0.12, 0.04, 0.008],         # monotone, ends < 0.01
        "mem_content": "X->Y under regime R; effect=0.31",
        "claimed_sig": hashlib.sha256("X->Y under regime R; effect=0.31".encode()).hexdigest()[:16],
        "action": {"cmd": "echo done", "path": "/sandbox/out.txt"},
    }


def run_scenario(name, ctx):
    spine = Spine([L0Physics(), L1Perception(), L2Causal(),
                   L3Decision(), L4Memory(), L5Execution()])
    sv = spine.run(ctx)
    if sv.is_answer():
        gaps = []
        for c in sv.credentials:
            _, info = c.verify()
            gaps.append(info.get("gap", 0.0))
        print(f"  {name:<26} -> SYSTEM ANSWER | 6/6 layers, max recompute gap = {max(gaps):.1e}")
    else:
        print(f"  {name:<26} -> SYSTEM ABSTAIN @ {sv.abstained_at}: {sv.reason}")
    return sv


def main():
    print("=== Fusion Phase A: the credential spine, end-to-end across L0..L5 ===\n")

    print("Scenario A — healthy input (every gate admissible, every credential recomputes):")
    a = run_scenario("healthy", healthy_ctx())

    print("\nScenario B — L0 fault (energy drift 5e-3 > 1e-3 threshold):")
    ctx_b = healthy_ctx(); ctx_b["energies"] = (1.0, 1.005)
    b = run_scenario("L0 energy drift", ctx_b)

    print("\nScenario C — a 'lying' L2 (claims do=0.30 but its own recompute yields 0.45):")
    ctx_c = healthy_ctx(); ctx_c["claimed_do"] = 0.30; ctx_c["do_method_b"] = 0.45
    c = run_scenario("L2 non-recomputable", ctx_c)

    ok = (a.is_answer() and a.decision is Decision.ANSWER
          and not b.is_answer() and b.abstained_at == "L0_physics"
          and not c.is_answer() and c.abstained_at == "L2_causal")
    print("\nSpine guarantees demonstrated:")
    print("  • healthy → ANSWER only when ALL 6 layers pass BOTH gates (admissible AND recomputable)")
    print("  • any single admissibility failure → ABSTAIN at that layer, nothing downstream runs")
    print("  • a confident-but-non-recomputable claim is auto-DOWNGRADED to ABSTAIN by the spine —")
    print("    the LLM failure mode (confident-narrow-wrong) is structurally barred, not hoped away")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
