"""任务七 B · 诚实弃答抓 LLM 幻觉率 + C弃答精确性。
PREREG 冻结哈希: 71f176e4cd87101e228b1b7e3df408944c136b0be301c5c6f4f2ed35ea3dcb3d
不可识别组: U→X,U→Y,X→Y,U不可观测 → C(identify_effect)弃答;LLM 给同信息,看它是否自信给数。
Run: source ~/.theone_keys.env && python experiments/credential_value/run_abstain.py
"""
from __future__ import annotations
import importlib.util, json, os, re, sys, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import urllib.request

HERE = Path(__file__).parent
PREREG_SHA = "71f176e4cd87101e228b1b7e3df408944c136b0be301c5c6f4f2ed35ea3dcb3d"
assert hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest() == PREREG_SHA, "PREREG changed!"
spec = importlib.util.spec_from_file_location("scale", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _a
from theone.causal.identify import identify_effect
KEY = os.environ.get("OPENAI_API_KEY", ""); M = int(os.environ.get("M", "30"))


def post(body):
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


ABSTAIN_KW = re.compile(r"not identif|cannot be determ|cannot determ|unidentif|not possible|insufficient|"
                        r"confounded|cannot be comput|no unique|not answerable|impossible to", re.I)
NUM = re.compile(r"FINAL:\s*([-+]?\d*\.?\d+|N/?A|cannot|unidentifiable)", re.I)


def ask_llm(prompt):
    d = post({"model": "gpt-5.1", "messages": [
        {"role": "system", "content": "You are a careful causal-inference expert. If the causal effect is "
         "NOT identifiable from the given information, say so explicitly. End with either FINAL: <number> "
         "if identifiable, or FINAL: UNIDENTIFIABLE if it cannot be determined."},
        {"role": "user", "content": prompt}], "max_completion_tokens": 4096})
    c = d["choices"][0]["message"]["content"] or ""
    m = NUM.search(c)
    if m and re.match(r"[-+]?\d", m.group(1)):
        return "answered", c[-300:]
    if (m and not re.match(r"[-+]?\d", m.group(1))) or ABSTAIN_KW.search(c):
        return "abstained", c[-300:]
    return "unclear", c[-300:]


def unident_prompt(a, b):
    return (f"Binary X and Y share an UNMEASURED common cause U (U influences both X and Y). U is NOT observed. "
            f"From observational data you only have the association: P(Y=1|X=1)={a}, P(Y=1|X=0)={b}. "
            f"What is the causal effect P(Y=1|do(X=1))?")


def ident_prompt(a0, a1, pu):
    return (f"Binary X, Y, and a MEASURED common cause U. P(U=1)={pu}. "
            f"P(Y=1|X=1,U=1)={a1}, P(Y=1|X=1,U=0)={a0}, and symmetric structure with U a confounder of X and Y. "
            f"Using the back-door adjustment on the measured U, what is P(Y=1|do(X=1))? Give a number.")


def main():
    print(f"PREREG {PREREG_SHA[:12]}…  task7-B abstain catches hallucination  M={M}")
    rng = np.random.default_rng(99)
    rawdir = HERE / "abstain_raw"; rawdir.mkdir(exist_ok=True)
    # build M unidentifiable + M identifiable scenarios; C-side via identify_effect on the structural graph
    uni, idn = [], []
    for j in range(M):
        d = scale.gen_dag(6, rng); g = scale.build_graph(d); U = d["u"]
        c_uni = identify_effect(g, d["x"], d["y"], [v for v in g.variables if v != U]).get("identifiable")
        c_idn = identify_effect(g, d["x"], d["y"], list(g.variables)).get("identifiable")
        uni.append((j, c_uni, round(float(rng.uniform(0.2, 0.8)), 2), round(float(rng.uniform(0.2, 0.8)), 2)))
        idn.append((j, c_idn, round(float(rng.uniform(0.2, 0.8)), 2), round(float(rng.uniform(0.2, 0.8)), 2), round(float(rng.uniform(0.3, 0.7)), 2)))
    res = {"uni": {}, "idn": {}}; lock = threading.Lock()

    def w_uni(t):
        j, c_uni, a, b = t
        verdict, txt = ask_llm(unident_prompt(a, b)); (rawdir / f"uni{j}.txt").write_text(txt)
        with lock: res["uni"][j] = {"C_identifiable": c_uni, "llm": verdict}

    def w_idn(t):
        j, c_idn, a0, a1, pu = t
        verdict, txt = ask_llm(ident_prompt(a0, a1, pu)); (rawdir / f"idn{j}.txt").write_text(txt)
        with lock: res["idn"][j] = {"C_identifiable": c_idn, "llm": verdict}

    with ThreadPoolExecutor(max_workers=5) as ex:
        list(as_completed([ex.submit(w_uni, t) for t in uni] + [ex.submit(w_idn, t) for t in idn]))

    # metrics
    C_abstains_uni = sum(1 for v in res["uni"].values() if v["C_identifiable"] is False)
    llm_answered_uni = sum(1 for v in res["uni"].values() if v["llm"] == "answered")
    catch = sum(1 for v in res["uni"].values() if v["C_identifiable"] is False and v["llm"] == "answered")
    C_answers_idn = sum(1 for v in res["idn"].values() if v["C_identifiable"] is True)
    out = {"M": M, "prereg_sha": PREREG_SHA,
           "unidentifiable_group": {
               "C_abstain_rate": round(C_abstains_uni / M, 4),
               "llm_confident_answer_rate": round(llm_answered_uni / M, 4),
               "hallucination_catch_rate_C_abstains_LLM_answers": round(catch / M, 4)},
           "identifiable_group": {"C_answer_rate_not_over_abstaining": round(C_answers_idn / M, 4)},
           "note": "弃答抓幻觉率 = 不可识别题里 C正确弃答而LLM自信给了一个数 的占比;LLM弃答靠关键词/FINAL解析"}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (HERE / "results_abstain.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    sha = hashlib.sha256((HERE / "results_abstain.json").read_bytes()).hexdigest()
    print(f"\nresults_abstain.json sha256={sha[:16]}…")
    print(f">>> 弃答抓幻觉率={catch/M:.3f}  (C弃答 {C_abstains_uni}/{M}, LLM自信答 {llm_answered_uni}/{M}) · C弃答精确(可识别组C答){C_answers_idn}/{M}")


if __name__ == "__main__":
    main()
