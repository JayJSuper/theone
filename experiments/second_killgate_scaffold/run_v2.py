"""第二命门 · 公平强脚手架 v2(直接链式推理 + 自洽 K=5,代码可选不强制)。
AMENDMENT-2 冻结哈希: df4ebce326de07fbdc907b42fa1a164e58c4ed0f98b01e27816757c2ea963d78
v1 失败脚手架(0.288)如实保留,本文件不改判据。

Run: source ~/.theone_keys.env && python experiments/second_killgate_scaffold/run_v2.py
"""
from __future__ import annotations
import importlib.util, json, os, re, subprocess, sys, tempfile, time, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).parent
AMEND_SHA = "df4ebce326de07fbdc907b42fa1a164e58c4ed0f98b01e27816757c2ea963d78"
assert hashlib.sha256((HERE / "PREREG_AMENDMENT2.md").read_bytes()).hexdigest() == AMEND_SHA, "AMENDMENT changed!"

spec = importlib.util.spec_from_file_location("scale_run", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _argv
N = int(os.environ.get("N", "150")); NODES, BASE_SEED, K = 12, 20260614, 5
KEY = os.environ.get("OPENAI_API_KEY", "")

import urllib.request
def post(body, timeout=300):
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

SYS = ("You are an expert at probabilistic reasoning over causal Bayesian networks. You will be given "
       "a network over binary variables and asked for P(Y=1 | do(X=1)). Reason step by step about the "
       "structure (parents, CPTs), apply the truncated factorization for the intervention, and compute "
       "the numeric probability. You may optionally write a ```python``` block (numpy/itertools, sandboxed) "
       "if you find it helpful, but it is not required. End with your final answer on its own line as "
       "exactly:  FINAL: <number>")
NUM = re.compile(r"FINAL:\s*([-+]?\d*\.?\d+)"); CODE = re.compile(r"```python\s*(.*?)```", re.S)

def run_code(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import numpy as np, itertools, math\n" + code); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=5, env={"PATH": "/usr/bin"})
        return (p.stdout or "")[-500:] + (("\n[stderr]" + p.stderr[-150:]) if p.stderr else "")
    except Exception as e:
        return f"[exec {e}]"
    finally:
        try: os.unlink(path)
        except Exception: pass

def once(text, rawdir, tag):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text + "\n\nWhat is P(Y=1 | do(X=1))?"}]
    tr = []
    for turn in range(4):
        d = post({"model": "gpt-5.1", "messages": msgs, "max_completion_tokens": 8192})
        c = d["choices"][0]["message"]["content"] or ""
        tr.append({"turn": turn, "content": c, "usage": d.get("usage")})
        m = NUM.search(c)
        if m:
            (rawdir / f"{tag}.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1))
            try: return float(m.group(1))
            except Exception: return None
        cm = CODE.search(c)
        msgs.append({"role": "assistant", "content": c})
        if cm:
            msgs.append({"role": "user", "content": f"Sandbox stdout:\n{run_code(cm.group(1))}\nNow give FINAL: <number>."})
        else:
            msgs.append({"role": "user", "content": "Give FINAL: <number> now."})
    (rawdir / f"{tag}.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1))
    return None

def answer(text, rawdir, i):
    vals = [v for k in range(K) if (v := once(text, rawdir, f"i{i}_k{k}")) is not None]
    return (float(np.median(vals)) if vals else None), len(vals)

def gen():
    rng = np.random.default_rng(BASE_SEED + NODES); out = []
    for i in range(N):
        d = scale.gen_dag(NODES, rng); g = scale.build_graph(d)
        truth = scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value
        out.append({"i": i, "text": scale.render_text(d), "truth": round(float(truth), 6)})
    return out

def main():
    print(f"AMENDMENT-2 verified {AMEND_SHA[:16]}…  N={N} K={K} (fair scaffold: direct CoT + self-consistency)")
    insts = gen(); rawdir = HERE / "scaffold_v2_raw"; rawdir.mkdir(exist_ok=True)
    cache = HERE / "rows_v2.jsonl"; done = {}
    if cache.exists():
        for ln in cache.read_text().splitlines():
            r = json.loads(ln)
            if r.get("scaffold_v2_pred") is not None: done[r["i"]] = r
    todo = [x for x in insts if x["i"] not in done]
    print(f"resume {len(done)} done, {len(todo)} to run")
    lock = threading.Lock(); t0 = time.time(); cnt = [0]
    def work(inst):
        sc, nv = answer(inst["text"], rawdir, inst["i"])
        row = {"i": inst["i"], "truth": inst["truth"], "scaffold_v2_pred": sc, "nvalid": nv}
        with lock:
            with open(cache, "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cnt[0] += 1
            print(f"  [{cnt[0]}/{len(todo)}] i={inst['i']:>3} v2={sc} valid={nv}/{K} truth={inst['truth']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in as_completed([ex.submit(work, x) for x in todo]): pass
    print("done scaffold-v2.", flush=True)

if __name__ == "__main__":
    main()
