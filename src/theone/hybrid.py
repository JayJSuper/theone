"""Hybrid pipeline v0 — the first real System1 + System2 closed loop (M1).

Division of labor (architecture review, frozen discipline):
  S2 (deterministic causal engine) COMPUTES — never the LLM.
  S1 (DeepSeek) UNDERSTANDS and EXPRESSES — never does causal math.
  Router v0 is rule-based and COST-SENSITIVE: suspected-causal queries must not
  fall through to S1 (a misroute there resurrects hallucination); when in doubt
  and no validated model exists, ABSTAIN.

Evidence-tiered graph library v0: every graph carries an evidence tier; only
graphs at tier >= "machine_validated" may serve do-queries. Demo graphs are
structurally validated against analytic truth but their parameters are
ILLUSTRATIVE — answers say so (honesty surfaces in the product, not just docs).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .types import Variable
from .causal.graph import CausalGraph
from .causal.engine import InterventionEngine
from .causal.identify import find_adjustment_set, identify_effect

# --------------------------------------------------------------------------
# Evidence-tiered SCM library v0
# --------------------------------------------------------------------------
TIERS = ("conjecture", "llm_extracted", "literature", "machine_validated")


@dataclass
class LibraryGraph:
    key: str
    treatment: str
    outcome: str
    confounder: str
    aliases: dict                      # var -> [keywords] (zh+en)
    evidence_tier: str
    note: str
    latent: tuple = ()                 # unobserved variables (cannot be adjusted on)
    graph: CausalGraph = field(repr=False, default=None)

    def usable_for_do(self) -> bool:
        return TIERS.index(self.evidence_tier) >= TIERS.index("machine_validated")


def identify_gate(lg: LibraryGraph) -> dict:
    """Identifiability gate (M1b + Q-C14): a do-query is answerable via backdoor,
    else front-door, else single-IV, among OBSERVED variables. The causal 'I
    don't know' is computable — when nothing identifies, name the missing
    variable instead of guessing. Front-door/IV carry unverifiable-assumption
    flags. Returns the unified identify_effect result, plus a 'missing' field on
    refusal for backward compatibility with the server."""
    observed = [v for v in lg.graph.variables if v not in set(lg.latent)]
    res = identify_effect(lg.graph, lg.treatment, lg.outcome, observed=observed)
    if not res["identifiable"]:
        res["missing"] = sorted(set(lg.latent)) or [lg.confounder]
    return res


def _confounded(u, x, y, p_u=0.5, px=(0.8, 0.2), py=(0.9, 0.5, 0.6, 0.2)):
    """Three-node confounded graph. py = P(y=1 | u,x) as (u1x1, u0x1, u1x0, u0x0).
    CPT keys are built against the ENGINE's actual parent_order (canonical, not
    insertion order) so variable naming can never silently permute parents."""
    g = CausalGraph()
    for n in (u, x, y):
        g.add_variable(Variable(n))
    g.add_edge(u, x); g.add_edge(u, y); g.add_edge(x, y)
    g.set_cpt(u, {(): {1: p_u, 0: 1 - p_u}})
    g.set_cpt(x, {(1,): {1: px[0], 0: 1 - px[0]}, (0,): {1: px[1], 0: 1 - px[1]}})
    by_uv = {(1, 1): py[0], (0, 1): py[1], (1, 0): py[2], (0, 0): py[3]}
    order = list(g.parent_order(y))               # e.g. (x, u) or (u, x)
    cpt_y = {}
    for (uv, xv), p1 in by_uv.items():
        key = tuple(uv if parent == u else xv for parent in order)
        cpt_y[key] = {1: p1, 0: 1 - p1}
    g.set_cpt(y, cpt_y)
    return g


def build_library() -> list:
    lib = [
        LibraryGraph(
            key="coffee_heart", treatment="coffee", outcome="heart_disease",
            confounder="smoking",
            aliases={"coffee": ["coffee", "咖啡"],
                     "heart_disease": ["heart", "心脏", "心臟"],
                     "smoking": ["smoking", "smoke", "吸烟", "抽烟"]},
            evidence_tier="machine_validated",
            note=("Structure machine-validated against analytic truth (T1 frozen "
                  "semantics); parameters are ILLUSTRATIVE, not real epidemiology."),
            graph=_confounded("smoking", "coffee", "heart_disease")),
        LibraryGraph(
            key="ads_sales", treatment="advertising", outcome="sales",
            confounder="season",
            aliases={"advertising": ["ad", "ads", "advertis", "广告", "投放"],
                     "sales": ["sales", "revenue", "销量", "销售", "营收"],
                     "season": ["season", "holiday", "季节", "旺季", "节日"]},
            evidence_tier="machine_validated",
            note=("Structure machine-validated; parameters illustrative — "
                  "marketing-textbook confounding (season drives both ads and sales)."),
            graph=_confounded("season", "advertising", "sales",
                              px=(0.7, 0.3), py=(0.8, 0.55, 0.5, 0.2))),
        LibraryGraph(
            key="sleep_hair", treatment="sleep_deprivation", outcome="hair_loss",
            confounder="stress",
            aliases={"sleep_deprivation": ["熬夜", "睡眠", "sleep", "stay up"],
                     "hair_loss": ["脱发", "掉发", "hair"],
                     "stress": ["压力", "焦虑", "stress"]},
            evidence_tier="machine_validated",
            latent=("stress",),          # stress is UNOBSERVED -> gate must refuse
            note=("Structure machine-validated; STRESS IS UNOBSERVED — the only "
                  "backdoor set is unavailable, so do-queries are unidentifiable. "
                  "This graph exists to demonstrate computable refusal."),
            graph=_confounded("stress", "sleep_deprivation", "hair_loss",
                              px=(0.75, 0.35), py=(0.7, 0.45, 0.5, 0.25))),
    ]
    return lib


# --------------------------------------------------------------------------
# Router v0 — rule-based, cost-sensitive (prefer abstain over misroute)
# --------------------------------------------------------------------------
_CAUSAL_PAT = re.compile(
    r"导致|因果|引起|造成|cause|causal|effect of|如果.{0,12}(会|将)|what if|"
    r"would .*有|do\(|干预|反事实|counterfactual", re.I)
_FORECAST_PAT = re.compile(
    r"明年|下个月|预测|会涨|会跌|涨到|跌到|next (year|month)|price.*will|"
    r"will.*price|forecast|比特币|bitcoin|股价|stock", re.I)
_MEM_PUT = re.compile(r"^\s*(remember|记住)[\s:：]+(.+)$", re.I | re.S)
_MEM_GET = re.compile(r"remember|记得|回忆|recall|上次|我告诉过", re.I)


def route(query: str, library: list) -> dict:
    """Returns {mode, graph?, payload?}. Modes: memory_put / memory_get /
    s2_causal / abstain_no_model / abstain_forecast / s1_direct."""
    m = _MEM_PUT.match(query)
    if m:
        return {"mode": "memory_put", "payload": m.group(2).strip()}
    causal = bool(_CAUSAL_PAT.search(query))
    hit = _match_graph(query, library)
    if causal or hit:
        if hit and hit.usable_for_do():
            return {"mode": "s2_causal", "graph": hit}
        if _FORECAST_PAT.search(query):
            return {"mode": "abstain_forecast"}
        return {"mode": "abstain_no_model"}      # causal intent, no validated model
    if _FORECAST_PAT.search(query):
        return {"mode": "abstain_forecast"}
    if _MEM_GET.search(query):
        return {"mode": "memory_get"}
    return {"mode": "s1_direct"}


def _match_graph(query: str, library: list):
    q = query.lower()
    best, best_n = None, 0
    for lg in library:
        vars_hit = sum(1 for v, kws in lg.aliases.items()
                       if any(k in q for k in kws))
        if vars_hit >= 2 and vars_hit > best_n:   # need >=2 of the graph's variables
            best, best_n = lg, vars_hit
    return best


# --------------------------------------------------------------------------
# S2 computation (deterministic; the only place causal numbers come from)
# --------------------------------------------------------------------------
def s2_answer(lg: LibraryGraph) -> dict:
    eng = InterventionEngine(lg.graph)
    X, Y = lg.treatment, lg.outcome
    obs1 = eng.query_observation(Y, 1, {X: 1}).value
    obs0 = eng.query_observation(Y, 1, {X: 0}).value
    int1 = eng.query_intervention(Y, 1, {X: 1}).value
    int0 = eng.query_intervention(Y, 1, {X: 0}).value
    adj = find_adjustment_set(lg.graph, X, Y)
    return {
        "treatment": X, "outcome": Y, "confounder": lg.confounder,
        "obs_given_x1": round(obs1, 6), "obs_given_x0": round(obs0, 6),
        "int_do_x1": round(int1, 6), "int_do_x0": round(int0, 6),
        "obs_ate": round(obs1 - obs0, 6), "int_ate": round(int1 - int0, 6),
        "confounding_bias": round((obs1 - obs0) - (int1 - int0), 6),
        "adjustment_set": sorted(adj) if adj is not None else None,
        "graph_hash": lg.graph.content_hash(),
        "evidence_tier": lg.evidence_tier, "note": lg.note,
        "method": "graph_surgery_do + exact_joint_conditioning (T1 frozen)",
    }


# --------------------------------------------------------------------------
# S1 expression (LLM renders S2's numbers; computes nothing)
# --------------------------------------------------------------------------
_S1_SYSTEM = (
    "You are S1, the language organ of The One. You are given EXACT numbers "
    "computed by a deterministic causal engine (S2). Express them clearly in the "
    "user's language. You MUST NOT change, recompute, or invent any number. "
    "Explain correlation-vs-causation using ONLY the provided figures. End with "
    "one sentence noting the stated evidence scope. Be concise (<=120 words).")


def s1_render(user_query: str, s2: dict, s1_client) -> str:
    """Render S2 numbers via the LLM; deterministic template fallback offline."""
    if s1_client is not None:
        try:
            prompt = (f"User asked: {user_query}\n\nS2 engine results "
                      f"(authoritative, do not alter): {s2}\n\n"
                      "Write the answer in the user's language.")
            out = s1_client.chat(
                [{"role": "system", "content": _S1_SYSTEM},
                 {"role": "user", "content": prompt}], max_tokens=400)
            txt = (out.get("content") or "").strip()
            if txt:
                return txt
        except Exception:
            pass                                  # graceful degrade to template
    return template_render(s2)


def template_render(s2: dict) -> str:
    return (
        f"Observed association: P({s2['outcome']}|{s2['treatment']}=1) = "
        f"{s2['obs_given_x1']:.2f} vs {s2['obs_given_x0']:.2f} (assoc. ATE "
        f"{s2['obs_ate']:+.2f}). After graph surgery do({s2['treatment']}): "
        f"{s2['int_do_x1']:.2f} vs {s2['int_do_x0']:.2f} (causal ATE "
        f"{s2['int_ate']:+.2f}). Confounding bias from {s2['confounder']}: "
        f"{s2['confounding_bias']:+.2f}. Adjustment set: {s2['adjustment_set']}. "
        f"Scope: {s2['note']}")
