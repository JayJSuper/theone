"""A-line demonstrable product · end-to-end demo through TheOneApp.

One app, four kinds of request, every answer with provenance:
  1. causal query  -> verifiable engine: exact do() + recomputable credential + E-value,
                      and the mounted LLM's number is corroborated/refuted (hallucination guard)
  2. memory store  -> sovereign causal memory
  3. memory recall -> sovereign causal memory
  4. chat/code     -> mounted LLM (live if key present), labelled UNVERIFIED

Registered domain: treatment T causally affects recovery R, confounded by severity S.

Run:  source ~/.theone_keys.env && .venv/bin/python experiments/aline_product/run.py
      (runs fully offline too — the LLM path degrades to a labelled stub)
"""
from __future__ import annotations
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.app import TheOneApp, CausalDomain


def recovery_domain():
    """S (severity) confounds T (treatment) and R (recovery); T also affects R."""
    g = CausalGraph()
    for n in ("S", "T", "R"):
        g.add_variable(Variable(n))
    g.add_edge("S", "T"); g.add_edge("S", "R"); g.add_edge("T", "R")
    g.set_cpt("S", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("T", {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.3, 1: 0.7}})  # sicker -> more treated
    oR = list(g.parent_order("R"))
    vals = {(0, 0): .80, (0, 1): .90, (1, 0): .30, (1, 1): .55}  # severity hurts, treatment helps
    g.set_cpt("R", {tuple(s if p == "S" else t for p in oR): {1: v, 0: round(1 - v, 2)}
                    for (s, t), v in vals.items()})
    aliases = {"treatment": "T", "treat": "T", "recovery": "R", "recover": "R",
               "severity": "S", "__treatment__": "T", "__target__": "R"}
    return CausalDomain("recovery", g, aliases)


def show(title, res):
    print(f"\n▸ {title}")
    print(f"   track={res['track']} · verified={res['verified']} · {res['provenance']}")
    print(f"   answer: {res['answer']}")
    if res.get("e_value") is not None:
        print(f"   credential: regime='{res['regime']}'")
        print(f"               recomputed_ok={res['recomputed_ok']} gap={res['recompute_gap']:.1e} "
              f"· E-value={res['e_value']}")
    if res.get("verdict"):
        print(f"   hallucination-guard: {res['verdict_note']}")
    if res.get("recent"):
        print(f"   recent memories: {res['recent']}")


def main():
    print("=== A-line demonstrable product · TheOneApp end-to-end ===")
    app = TheOneApp(provider="deepseek", domain=recovery_domain())
    live = app.llm.available()
    print(f"mounted LLM: deepseek · live={live}")

    r1 = app.ask("What is the causal effect of the treatment on recovery?")
    show("causal query (verifiable)", r1)

    r2 = app.ask("Please remember that the treatment helps recovery once severity is adjusted for.")
    show("memory store (sovereign)", r2)

    r3 = app.ask("recall what we stored")
    show("memory recall (sovereign)", r3)

    r4 = app.ask("Write a one-line Python function that returns the square of a number.")
    show("code generation (mounted LLM, unverified)", r4)

    # self-test (works offline too)
    ok = (r1["verified"] is True and r1.get("value") is not None and r1["recomputed_ok"] is True
          and r2["verified"] is True and "remembered" in r2["answer"]
          and r3["verified"] is True
          and r4["verified"] is False and "UNVERIFIED" in r4["provenance"])
    if live and r1.get("verdict"):
        ok = ok and r1["verdict"] in ("corroborated", "refuted")
    app.close()
    print("\nprovenance contract: causal answers are engine-verified with a recomputable")
    print("credential; the mounted LLM's number is corroborated/refuted; generation is")
    print("clearly labelled UNVERIFIED. One product, two engines, every answer's source known.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
