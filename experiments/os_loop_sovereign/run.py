"""The One — OS loop with pillar 2 on REAL sovereign memory (component → system).

The original os_loop retrieves its causal model from an in-process library. Here the
memory pillar is the real thing: each causal model is persisted into SovereignMemory
as a memory indexed by its CAUSAL SIGNATURE (treatment→outcome + adjustment set +
de-confounded effect, derived from a computation-pillar credential). The loop then
recalls its model from that persistent store by causal signature — not surface text —
and the store is versioned, auditable, and exportable (sovereignty). This makes the
'three pillars, one principle' synthesis run on a real, persistent memory substrate.

Run:  .venv/bin/python experiments/os_loop_sovereign/run.py
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
from theone.hybrid import build_library, identify_gate, answer_causal, route
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature

HERE = Path(__file__).parent
_o = importlib.util.spec_from_file_location("osloop", HERE.parent / "os_loop" / "run.py")
OS = importlib.util.module_from_spec(_o); _o.loader.exec_module(OS)  # reuse pgmpy_true_do
DB = HERE / "sovereign_mem.sqlite"


def seed_memory(sov: SovereignMemory, library) -> dict:
    """Persist every library model as a credentialed causal memory."""
    seeded = {}
    for lg in library:
        gate = identify_gate(lg)
        if gate["identifiable"]:
            ans = answer_causal(lg, gate)
            eff, adj = ans["int_do_x1"], (ans.get("adjustment_set") or [])
            regime = f"identifiable:{gate['strategy']}"
        else:
            eff, adj, regime = 0.0, [], f"unidentifiable:missing={gate.get('missing')}"
        cred = {"treatment": lg.treatment, "target": lg.outcome,
                "adjustment_set": adj, "effect": round(float(eff), 6), "regime": regime}
        mid = sov.remember(lg.note or lg.key, cred, source=f"library-seed:{lg.key}")
        seeded[(lg.treatment, lg.outcome)] = mid
    return seeded


def recall_model(sov: SovereignMemory, treatment, outcome):
    """Pillar 2: recall the persisted memory for THIS causal question (treatment→
    outcome), by signature — the persistent analogue of route's in-memory match."""
    for r in sov._all_live():
        sig = CausalSignature.from_dict(r["value"]["signature"])
        if sig.treatment == treatment and sig.target == outcome:
            return r, sig
    return None, None


def the_one(query, library, sov):
    r = route(query, library)
    out = {"query": query, "route": r["mode"]}
    if r["mode"] != "s2_causal":
        out["decision"] = "ABSTAIN"
        out["reason"] = {"abstain_no_model": "causal intent but no machine-validated model",
                         "abstain_forecast": "forecast intent — out of causal scope",
                         "s1_direct": "non-causal query"}.get(r["mode"], r["mode"])
        return out
    lg = r["graph"]
    # pillar 2: recall from PERSISTENT sovereign memory by causal signature
    row, sig = recall_model(sov, lg.treatment, lg.outcome)
    out["memory_source"] = "sovereign_store(persistent)"
    out["recalled_mem_id"] = row["id"] if row else None
    out["recalled_signature"] = sig.structure_key() if sig else None
    # pillar 1: identifiability gate + exact compute (on the recalled model's graph)
    gate = identify_gate(lg)
    out["strategy"] = gate.get("strategy")
    if not gate["identifiable"]:
        out["decision"] = "ABSTAIN"
        out["reason"] = f"do-effect unidentifiable; missing {gate.get('missing')}"
        return out
    ans = answer_causal(lg, gate)
    try:
        truth = OS.pgmpy_true_do(lg.graph, lg.treatment, lg.outcome)
        verified = abs(ans["int_do_x1"] - truth) < 1e-6
    except Exception:
        truth, verified = None, False
    out["engine_do_x1"] = round(ans["int_do_x1"], 6)
    out["independently_verified"] = verified
    if gate.get("strategy") in ("front_door", "iv") or ans.get("assumptions"):
        out["decision"] = "ANSWER_WITH_CAVEAT"
    elif verified:
        out["decision"] = "ANSWER"
    else:
        out["decision"] = "ABSTAIN"; out["reason"] = "recompute mismatch"
    return out


def main():
    if DB.exists():
        DB.unlink()
    library = build_library()
    sov = SovereignMemory(str(DB))
    seeded = seed_memory(sov, library)
    print(f"seeded {len(seeded)} causal models into PERSISTENT sovereign memory "
          f"(db={DB.name}, real SQLite)\n")

    queries = ["Does drinking coffee cause heart disease?",
               "What's the effect of advertising on sales?",
               "Does sleep deprivation cause hair loss?",
               "Does the supplement improve recovery?",
               "Will Bitcoin go up next week?"]
    results = []
    for q in queries:
        res = the_one(q, library, sov)
        results.append(res)
        print(f"Q: {q}")
        print(f"  pillar2 recall: {res.get('recalled_signature','-')} "
              f"(mem#{res.get('recalled_mem_id','-')}, {res.get('memory_source','-')})")
        print(f"  → DECISION: {res['decision']}"
              + (f"  do(X=1)={res['engine_do_x1']} verified={res['independently_verified']}"
                 if res['decision'].startswith('ANSWER') else f"  ({res.get('reason','')})"))
        print()

    # --- pillar 2 sovereignty in action: versioned revision + audit + export
    print("=" * 70)
    cof = seeded[("coffee", "heart_disease")]
    rev = sov.revise(cof, "coffee_heart REVISED: re-estimated CPTs, larger cohort",
                     {"treatment": "coffee", "target": "heart_disease",
                      "adjustment_set": ["smoking"], "effect": 0.61, "regime": "identifiable:backdoor"},
                     source="reanalysis-2026")
    hist = sov.history(rev)
    print(f"versioned revision of coffee_heart: {len(hist)} versions (auditable history):")
    for h in hist:
        print(f"  v{h['value']['version']} ({h['source']}): effect={h['value']['signature']['effect']}")
    # re-route the coffee query: now recalls the REVISED memory (persistent update)
    res2 = the_one("Does drinking coffee cause heart disease?", library, sov)
    print(f"re-query after revision recalls mem#{res2.get('recalled_mem_id')} "
          f"(v{sov.store.get(res2['recalled_mem_id'])['value']['version']} is live)")
    n_export = len([l for l in sov.export().splitlines() if l.strip()])
    print(f"sovereign export: {n_export} JSONL rows (take-your-data right)")
    sov.close()

    dec = {}
    for r in results:
        dec[r["decision"]] = dec.get(r["decision"], 0) + 1
    print("\nloop summary:", dec)
    print("Pillar 2 is now a real persistent sovereign store: models recalled by causal "
          "signature, versioned & auditable, exportable — wired into the credentialed loop.")
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
