"""B2 toward fluent language — CAUSAL-MASKED graph-to-text (DeepSeek's idea, made real).

DeepSeek's reframe: don't make an LLM honest — make the VERIFIED STRUCTURE speak. A decoder
renders a verified causal result into text, but its causal-claim slots are ARCHITECTURALLY
MASKED to the verified facts — it is structurally incapable of asserting a relation/direction/
magnitude that isn't in the credential. We also LEARN the phrasing STYLE (assertive vs hedged)
conditioned on the verification zone, and get phrasing DIVERSITY via the find/cover principle.

What we prove:
  1. LEARNED conditioning: style adapts to the zone (VERIFIABLE -> assertive "does/strong";
     UNCERTAIN -> hedged "may/might"; REJECT -> abstain). Trained, not hand-coded per call.
  2. ARCHITECTURAL no-hallucination: across many structures x phrasings, EVERY causal claim
     parsed back equals the verified fact (direction, magnitude, entities). 0 hallucinations.
  3. CONTRAST: an UNCONSTRAINED free generator (samples plausible causal sentences not tied to
     the structure) hallucinates — the parser catches it. The masked decoder cannot.

Honest scope: controlled phrasing vocabulary (not open-domain prose); the win is the typed-slot
architectural guarantee + learned style + diversity. Open-domain learned fluency (a real text
corpus trained masked decoder) is the next step.

Run:  .venv/bin/python experiments/bline_b2_graph2text/run.py
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cpu")
torch.manual_seed(0)
rng = np.random.default_rng(0)

DIRWORD = {1: "increases", -1: "decreases"}                  # masked: only the verified direction
MAGWORD = {0: "slightly", 1: "moderately", 2: "strongly"}    # masked: only the verified magnitude
# phrasing templates, indexed by learned STYLE (0=assertive, 1=hedged). {d}=dir {m}=mag slots.
TEMPLATES = {
    0: ["The treatment {m} {d} the outcome.",
        "Adjusting for confounders, the treatment {m} {d} the outcome.",
        "We verify the treatment {m} {d} the outcome."],
    1: ["The treatment may {m} {d} the outcome, but the evidence is uncertain.",
        "There is weak, unverified indication the treatment might {d} the outcome.",
        "The treatment could {m} {d} the outcome — not certain."],
}
ABSTAIN = "The effect cannot be verified, so no causal claim is made."


# --- learned style selector: zone+magnitude -> style (assertive/hedged) ----------------------
class StyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def struct_feats(zone_id, mag):
    # zone_id: 0=VERIFIABLE 1=UNCERTAIN 2=REJECT ; mag 0..2
    return torch.tensor([float(zone_id == 0), float(zone_id == 1), mag / 2.0], dtype=torch.float32)


def train_style(steps=1500):
    """Learn: VERIFIABLE -> assertive(0); UNCERTAIN -> hedged(1). (REJECT handled by abstain.)"""
    m = StyleNet(); opt = torch.optim.Adam(m.parameters(), lr=5e-3); lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        zone = int(rng.integers(0, 2)); mag = int(rng.integers(0, 3))
        x = struct_feats(zone, mag).unsqueeze(0)
        y = torch.tensor([0 if zone == 0 else 1])              # the style the corpus uses
        opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
    return m


def render(structure, style_net, n=1):
    """Graph-to-text with MASKED causal slots. structure = dict(direction, magnitude, zone_id)."""
    if structure["zone_id"] == 2:                              # REJECT -> abstain, no claim
        return [ABSTAIN]
    with torch.no_grad():
        style = int(style_net(struct_feats(structure["zone_id"], structure["magnitude"])
                              .unsqueeze(0)).argmax(1))
    d = DIRWORD[structure["direction"]]                        # <-- MASK: only the verified fact
    mword = MAGWORD[structure["magnitude"]]                    # <-- MASK: only the verified fact
    outs = []
    for _ in range(n):
        t = TEMPLATES[style][int(rng.integers(len(TEMPLATES[style])))]
        outs.append(t.format(m=mword, d=d))
    return outs


def parse_claim(sentence):
    """Parse the causal claim back out: (direction, magnitude) asserted, or None if abstain."""
    if "cannot be verified" in sentence:
        return None
    dirw = next((k for k, w in DIRWORD.items() if w in sentence), None)
    magw = next((k for k, w in MAGWORD.items() if w in sentence), None)
    return (dirw, magw)


def free_generator(structure):
    """UNCONSTRAINED baseline: samples a plausible causal sentence NOT tied to the structure
    (like an LLM riffing) -> can assert the WRONG direction/magnitude = hallucination."""
    d = DIRWORD[int(rng.choice([1, -1]))]
    mword = MAGWORD[int(rng.integers(0, 3))]
    return f"The treatment {mword} {d} the outcome."


def main():
    print("=== B2 toward language · causal-MASKED graph-to-text (verified structure speaks) ===\n")
    style_net = train_style()

    # learned-conditioning demo
    print("learned style conditioning (same fact, different zone):")
    for zid, name in [(0, "VERIFIABLE"), (1, "UNCERTAIN"), (2, "REJECT")]:
        s = {"direction": 1, "magnitude": 2, "zone_id": zid}
        print(f"  {name:11}: {render(s, style_net)[0]}")

    # architectural no-hallucination across many structures x phrasings
    halluc_masked = 0; total = 0; styles_seen = {0: set(), 1: set()}
    for _ in range(300):
        s = {"direction": int(rng.choice([1, -1])), "magnitude": int(rng.integers(0, 3)),
             "zone_id": int(rng.integers(0, 2))}                # verifiable/uncertain
        for sent in render(s, style_net, n=3):
            total += 1
            pd, pm = parse_claim(sent) or (None, None)
            # hallucination = ASSERTING a fact CONTRARY to the verified one. Omitting a slot
            # (None) is NOT a hallucination — the masked decoder can only fill slots with
            # verified values, so an asserted value is always correct.
            if (pd is not None and pd != s["direction"]) or (pm is not None and pm != s["magnitude"]):
                halluc_masked += 1
            styles_seen[0 if s["zone_id"] == 0 else 1].add(sent)

    # contrast: free generator on the same structures
    halluc_free = 0
    for _ in range(300):
        s = {"direction": int(rng.choice([1, -1])), "magnitude": int(rng.integers(0, 3))}
        sent = free_generator(s)
        pd, pm = parse_claim(sent) or (None, None)
        if (pd is not None and pd != s["direction"]) or (pm is not None and pm != s["magnitude"]):
            halluc_free += 1

    # style conditioning correctness
    v = {"direction": 1, "magnitude": 1, "zone_id": 0}; u = {"direction": 1, "magnitude": 1, "zone_id": 1}
    assertive_ok = "may" not in render(v, style_net)[0] and "uncertain" not in render(v, style_net)[0]
    hedged_ok = any(w in render(u, style_net)[0] for w in ("may", "might", "could", "uncertain"))

    print(f"\nmasked decoder: {halluc_masked}/{total} hallucinations across structures x phrasings")
    print(f"free baseline : {halluc_free}/300 hallucinations (asserts unverified direction/magnitude)")
    print(f"phrasing diversity: assertive {len(styles_seen[0])} variants · hedged {len(styles_seen[1])} variants")

    g1 = halluc_masked == 0                                   # architecturally cannot hallucinate
    g2 = halluc_free > 60                                     # free generator DOES hallucinate (caught)
    g3 = assertive_ok and hedged_ok                          # learned style conditioning works
    g4 = len(styles_seen[0]) >= 3 and len(styles_seen[1]) >= 3   # phrasing diversity
    allok = g1 and g2 and g3 and g4
    print("\ngraph2text gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] masked decoder: 0 hallucinated causal claims (architectural)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] unconstrained free generator hallucinates (parser catches it)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] LEARNED style conditions on zone (assertive vs hedged)")
    print(f"  [{'PASS' if g4 else 'FAIL'}] phrasing diversity (>=3 variants per style)")
    print(f"\n  >>> {'PASS — verified structure speaks: learned style, diverse phrasing, architecturally cannot hallucinate' if allok else 'CHECK'}")
    print("\nHonest: controlled phrasing vocabulary, not open-domain prose. The win is the typed-slot")
    print("ARCHITECTURAL guarantee (the causal claim can only be a verified fact) + LEARNED style +")
    print("diversity. Open-domain learned fluency (a masked decoder trained on a real text corpus,")
    print("DeepSeek's full proposal) is the next step. The principle holds: the structure speaks,")
    print("the model cannot invent a relation it cannot recompute.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
