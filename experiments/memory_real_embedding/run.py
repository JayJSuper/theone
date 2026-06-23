"""Pillar 2 (sovereign memory) — REAL-EMBEDDING replication of the causal-cliff
finding (experiments/memory_causal_cliff). That experiment was pure simulation:
"association" a_m and "causal signature" c_m were scalars, and credential retrieval
matched on a noisy de-confounded scalar. The science claim was:

  * flat (cosine / surface-similarity) retrieval matches on ASSOCIATION -> under
    confounding it systematically returns the wrong memory and transfers the wrong
    causal effect (error grows with confounding σ).
  * credentialed (causal-signature) retrieval matches on the de-confounded effect ->
    error is INDEPENDENT of confounding (a flat credential-noise floor).

Here we test whether the same thing happens when "surface similarity" is a REAL
text embedding (OpenAI text-embedding-3-small) over synthetic memory texts, instead
of a scalar association.

DESIGN
------
Each memory is a past intervention "treatment T_k in environment E" with:
  - c   : TRUE de-confounded (back-door adjusted) causal effect, c ~ U(-2, 2).
  - a   : OBSERVED / confounded effect, a = c + N(0, σ). This is what naive
          observation reports; the confounder (environment severity) shifts it.
  - text: a natural-language clinical-style note. CRUCIAL: the wording describes the
          OBSERVED outcome magnitude a (binned into qualitative language), because
          that is what a surface report records. The text does NOT reveal c.
  - credential signature c (carried out-of-band, as if produced by Pillar 1's
          de-confounding computation), available only with credential noise 0.1.

A query is a fresh analogous intervention with true effect c_q; we observe its
confounded report a_q (-> query text) and, via its credential, ĉ_q = c_q + N(0,0.1).
We retrieve top-1 from the bank and transfer its c:
  - flat      : nearest memory by COSINE of real text embeddings (query text vs bank).
  - credential: nearest memory by |c - ĉ_q| (de-confounded signature match).
Error = |c_retrieved - c_q|, the error in the transferred causal effect.

If real embeddings track the confounded surface magnitude a (as wording does), flat
retrieval should reproduce the simulation: fine at σ=0, failing as σ grows; credential
retrieval should stay flat. If embeddings instead recover c somehow, flat would NOT
fail — that would refute the sim and is reported honestly.

Bounded: tiny corpus, results + embeddings cached to disk (resumable), <=50 API calls
budget enforced. Run:  source ~/.theone_keys.env && .venv/bin/python run.py
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "embed_cache.json"          # text -> embedding vector (resume / no re-pay)
RESULTS = HERE / "results.json"
EMBED_MODEL = "text-embedding-3-small"
EMBED_URL = "https://api.openai.com/v1/embeddings"
MAX_API_CALLS = 50                          # hard budget guard

# --- corpus knobs (kept small on purpose) ---
M = 24                      # memories in the bank (per confounding level)
N_QUERIES = 6               # queries evaluated per confounding level
CONF = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]     # confounding σ, same grid as the sim
SEED = 20260617

# qualitative outcome bins keyed by observed effect magnitude (this is what gets
# embedded — wording tracks the OBSERVED/confounded effect a, not the true c).
def effect_phrase(a: float) -> str:
    sign = "improved" if a >= 0 else "worsened"
    m = abs(a)
    if m < 0.25:   mag = "negligibly"
    elif m < 0.6:  mag = "slightly"
    elif m < 1.1:  mag = "moderately"
    elif m < 1.7:  mag = "substantially"
    else:          mag = "dramatically"
    return f"{mag} {sign}"

TREATMENTS = [
    "low-dose compound A", "the standard care protocol", "a short course of compound B",
    "the combined regimen", "an adjusted dosage of compound A", "the experimental therapy",
    "supportive treatment only", "the high-intensity protocol",
]
ENVIRONMENTS = [
    "an outpatient cohort", "a tertiary referral ward", "a community clinic",
    "a high-acuity unit", "a routine follow-up clinic", "a mixed-severity cohort",
]

def memory_text(rng, a: float) -> str:
    t = TREATMENTS[rng.integers(len(TREATMENTS))]
    e = ENVIRONMENTS[rng.integers(len(ENVIRONMENTS))]
    phrase = effect_phrase(a)
    # Surface note: describes the OBSERVED outcome only. Deliberately uniform template
    # so the embedding's main moving part is the observed-magnitude wording (= a).
    return (f"Clinical note: {t} was administered in {e}. "
            f"On follow-up, patient outcomes {phrase} relative to baseline.")

def query_text(rng, a_q: float) -> str:
    t = TREATMENTS[rng.integers(len(TREATMENTS))]
    e = ENVIRONMENTS[rng.integers(len(ENVIRONMENTS))]
    phrase = effect_phrase(a_q)
    return (f"Clinical note: {t} was administered in {e}. "
            f"On follow-up, patient outcomes {phrase} relative to baseline.")


# ---------------- embedding with disk cache + budget guard ----------------
class Embedder:
    def __init__(self):
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.api_calls = 0
        self.key = os.environ.get("OPENAI_API_KEY")

    def _save(self):
        CACHE.write_text(json.dumps(self.cache))

    def embed(self, texts: list[str]) -> np.ndarray:
        out, missing = {}, []
        for t in texts:
            if t in self.cache:
                out[t] = self.cache[t]
            else:
                missing.append(t)
        # de-dup missing
        missing = list(dict.fromkeys(missing))
        if missing:
            if not self.key:
                raise RuntimeError("OPENAI_API_KEY not set; source ~/.theone_keys.env")
            # batch all missing in ONE call (counts as 1 API call)
            if self.api_calls >= MAX_API_CALLS:
                raise RuntimeError(f"API budget {MAX_API_CALLS} exhausted")
            vecs = self._call(missing)
            self.api_calls += 1
            for t, v in zip(missing, vecs):
                self.cache[t] = v
                out[t] = v
            self._save()
        return np.array([out[t] for t in texts], dtype=np.float64)

    def _call(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
        req = urllib.request.Request(
            EMBED_URL, data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read())
                items = sorted(data["data"], key=lambda d: d["index"])
                return [it["embedding"] for it in items]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 3:
                    time.sleep(2 ** attempt); continue
                raise
        raise RuntimeError("embedding call failed after retries")


def cos_sim(q: np.ndarray, bank: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-12)
    bn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
    return bn @ qn


def build_level(rng, conf_std):
    """Build bank + queries for one confounding level. Returns texts + arrays."""
    c = rng.uniform(-2, 2, M)
    a = c + rng.normal(0, conf_std, M)
    mem_texts = [memory_text(rng, a[i]) for i in range(M)]

    cq = rng.uniform(-2, 2, N_QUERIES)
    aq = cq + rng.normal(0, conf_std, N_QUERIES)
    chat = cq + rng.normal(0, 0.10, N_QUERIES)          # credentialed de-confounded est.
    q_texts = [query_text(rng, aq[i]) for i in range(N_QUERIES)]
    return dict(c=c, a=a, mem_texts=mem_texts, cq=cq, aq=aq, chat=chat, q_texts=q_texts)


def main():
    rng = np.random.default_rng(SEED)
    emb = Embedder()
    rows = {}
    detail = {}
    for cs in CONF:
        d = build_level(rng, cs)
        bank_vecs = emb.embed(d["mem_texts"])           # cached + budgeted
        q_vecs = emb.embed(d["q_texts"])
        ef, ec, ea = [], [], []
        flat_picks = []
        for j in range(N_QUERIES):
            sims = cos_sim(q_vecs[j], bank_vecs)
            i_flat = int(np.argmax(sims))               # nearest by REAL embedding cosine
            err_flat = abs(d["c"][i_flat] - d["cq"][j])
            i_cred = int(np.argmin(np.abs(d["c"] - d["chat"][j])))   # de-confounded match
            err_cred = abs(d["c"][i_cred] - d["cq"][j])
            # diagnostic: error vs the ORACLE association baseline (nearest by true a),
            # i.e. the clean-scalar baseline the SIM assumed flat retrieval achieves.
            i_a = int(np.argmin(np.abs(d["a"] - d["aq"][j])))
            ea.append(abs(d["c"][i_a] - d["cq"][j]))
            ef.append(err_flat); ec.append(err_cred)
            flat_picks.append({"i_flat": i_flat, "i_cred": i_cred,
                               "cq": round(float(d["cq"][j]), 3),
                               "c_flat": round(float(d["c"][i_flat]), 3),
                               "c_cred": round(float(d["c"][i_cred]), 3)})
        rows[cs] = {"flat_mae": round(float(np.mean(ef)), 4),
                    "cred_mae": round(float(np.mean(ec)), 4),
                    "oracle_assoc_mae": round(float(np.mean(ea)), 4),
                    "n_q": N_QUERIES}
        detail[cs] = flat_picks

    out = {"model": EMBED_MODEL, "M": M, "N_QUERIES": N_QUERIES, "seed": SEED,
           "rows": rows, "api_calls_used": emb.api_calls, "detail": detail}
    RESULTS.write_text(json.dumps(out, indent=2))

    print(f"REAL-EMBEDDING memory retrieval for causal transfer ({EMBED_MODEL})")
    print(f"bank M={M}, queries/level={N_QUERIES}, API calls used={emb.api_calls}\n")
    print(f"{'conf σ':>8} | {'flat(embed cosine) MAE':>22} | {'credential MAE':>14} | {'oracle-a MAE':>12} | {'flat/cred':>9}")
    for cs in CONF:
        f, c = rows[cs]["flat_mae"], rows[cs]["cred_mae"]
        o = rows[cs]["oracle_assoc_mae"]
        print(f"{cs:>8.2f} | {f:>22.4f} | {c:>14.4f} | {o:>12.4f} | {f/max(c,1e-9):>8.1f}x")
    print("\n(see results.json for per-query retrieval detail)")


if __name__ == "__main__":
    main()
