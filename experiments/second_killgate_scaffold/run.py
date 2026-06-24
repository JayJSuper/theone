"""第二命门 · 三臂对照(C vs 裸 gpt-5.1 vs gpt-5.1+脚手架)。
PREREG 冻结哈希(跑前计算,不改判据): de7b328dfdc1aa8bd945c709d4001971e3f9442915e1dacde2a308b52e58c618
判据见 PREREG.md。脚手架 = CoT + 工具调用(模型写 Python 做后门调整,我方隔离执行)+ K=5 自洽中位数。

Run: source ~/.theone_keys.env && python experiments/second_killgate_scaffold/run.py
     N=8 python ... 可先小样验证(默认 150)。
"""
from __future__ import annotations
import importlib.util, json, os, re, subprocess, sys, tempfile, time, urllib.request, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
PREREG_SHA = "de7b328dfdc1aa8bd945c709d4001971e3f9442915e1dacde2a308b52e58c618"
# verify the prereg file still hashes to the frozen value (discipline: judged criteria unchanged)
_actual = hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest()
assert _actual == PREREG_SHA, f"PREREG changed! {_actual} != {PREREG_SHA}"

# ---- reuse the EXACT Baseline B instance generator + engine + scoring ----
spec = importlib.util.spec_from_file_location("scale_run", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _argv
TOL = 0.005
N = int(os.environ.get("N", "150"))
NODES, BASE_SEED, K = 12, 20260614, 5
KEY = os.environ.get("OPENAI_API_KEY", "")


def gen_l_tier():
    rng = np.random.default_rng(BASE_SEED + NODES); out = []
    for i in range(N):
        d = scale.gen_dag(NODES, rng); g = scale.build_graph(d)
        truth = scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value
        out.append({"i": i, "text": scale.render_text(d), "truth": round(float(truth), 6),
                    "C": round(float(truth), 6)})   # C = cognitive engine = exact (deterministic), by construction == truth
    return out


def post(body, timeout=300):
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


SCAFFOLD_SYS = (
    "You are a careful causal-inference assistant. You are given a description of a causal Bayesian "
    "network over binary variables and asked for P(Y=1 | do(X=1)). Reason step by step about the "
    "structure, the parents/CPTs, and the correct back-door / truncated-factorization computation. "
    "You MAY (and should) write ONE Python code block (```python ... ```) using only numpy/itertools "
    "to compute the exact interventional probability by enumeration; it will be executed in a sandbox "
    "and its stdout returned to you. After you have the computed value, output your final answer on its "
    "own line as exactly:  FINAL: <number>")


def run_code(code):
    """execute model-written code in an isolated subprocess (no network, 5s, only stdout)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import numpy as np, itertools, math\n" + code); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=5,
                           env={"PATH": "/usr/bin"})
        return (p.stdout or "")[-500:] + (("\n[stderr]" + p.stderr[-200:]) if p.stderr else "")
    except subprocess.TimeoutExpired:
        return "[timeout]"
    except Exception as e:
        return f"[exec-error {e}]"
    finally:
        try: os.unlink(path)
        except Exception: pass


NUM = re.compile(r"FINAL:\s*([-+]?\d*\.?\d+)")
CODE = re.compile(r"```python\s*(.*?)```", re.S)


def scaffold_once(text, rawdir, tag):
    """one CoT+tool-use pipeline pass -> a numeric answer or None (protocol failure)."""
    msgs = [{"role": "system", "content": SCAFFOLD_SYS},
            {"role": "user", "content": text + "\n\nWhat is P(Y=1 | do(X=1))?"}]
    transcript = []
    for turn in range(4):                      # up to 4 turns (code rounds + final)
        d = post({"model": "gpt-5.1", "messages": msgs, "max_completion_tokens": 8192})
        content = d["choices"][0]["message"]["content"] or ""
        transcript.append({"turn": turn, "content": content, "usage": d.get("usage")})
        m = NUM.search(content)
        if m:
            (rawdir / f"{tag}.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
            try: return float(m.group(1))
            except Exception: return None
        cm = CODE.search(content)
        if cm:
            out = run_code(cm.group(1))
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": f"Sandbox stdout:\n{out}\nNow give FINAL: <number>."})
        else:
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": "Give FINAL: <number> now (write code first if needed)."})
    (rawdir / f"{tag}.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
    return None


def scaffold_answer(text, rawdir, i):
    """K self-consistency passes -> median of numeric answers; None if all fail."""
    vals = []
    for k in range(K):
        v = scaffold_once(text, rawdir, f"i{i}_k{k}")
        if v is not None: vals.append(v)
    return (float(np.median(vals)) if vals else None), len(vals)


def main():
    print(f"PREREG hash verified: {PREREG_SHA[:16]}…  N={N} K={K}")
    insts = gen_l_tier()
    # bare gpt-5.1 per-instance from the frozen crossbase run (same 150 instances)
    bare = {}
    cb = HERE.parent / "baseline_b_crossbase" / "rows.jsonl"
    if cb.exists():
        for ln in cb.read_text().splitlines():
            r = json.loads(ln)
            if "gpt-5.1" in str(r).lower() or "gpt5" in str(r).lower() or r.get("base") == "gpt-5.1":
                bare[r.get("i")] = r
    rawdir = HERE / "scaffold_raw"; rawdir.mkdir(exist_ok=True)
    cache = HERE / "rows.jsonl"; done = {}
    if cache.exists():
        for ln in cache.read_text().splitlines():
            r = json.loads(ln)
            if r.get("scaffold_pred") is not None: done[r["i"]] = r
    todo = [x for x in insts if x["i"] not in done]
    print(f"  resume: {len(done)} done, {len(todo)} to run (parallel)", flush=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock(); t0 = time.time(); cnt = [0]

    def work(inst):
        sc, nvalid = scaffold_answer(inst["text"], rawdir, inst["i"])
        row = {"i": inst["i"], "truth": inst["truth"], "C": inst["C"],
               "scaffold_pred": sc, "scaffold_nvalid": nvalid}
        with lock:
            with open(cache, "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cnt[0] += 1
            print(f"  [{cnt[0]}/{len(todo)}] i={inst['i']:>3} scaffold={sc} valid={nvalid}/{K} "
                  f"truth={inst['truth']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in as_completed([ex.submit(work, x) for x in todo]):
            pass
    print("done scaffold arm.", flush=True)


if __name__ == "__main__":
    main()
