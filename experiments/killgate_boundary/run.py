"""任务五 · 第二命门边界扫描。规模轴{6,12,16} × 对手轴{bare,sc5,sc15,sc_tools}。
PREREG 冻结哈希: a77c9d7204b88979f473ecd0b9e0872b5bc36300db3e9acf6a7cfe9d3ccfed75
Run: source ~/.theone_keys.env && python experiments/killgate_boundary/run.py
"""
from __future__ import annotations
import importlib.util, json, os, re, subprocess, sys, tempfile, time, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = Path(__file__).parent
PREREG_SHA = "a77c9d7204b88979f473ecd0b9e0872b5bc36300db3e9acf6a7cfe9d3ccfed75"
assert hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest() == PREREG_SHA, "PREREG changed!"
spec = importlib.util.spec_from_file_location("scale", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _a
KEY = os.environ.get("OPENAI_API_KEY", "")
SIZES = [6, 12, 16]; N = int(os.environ.get("N", "24")); BASE_SEED = 20260624
ADVERSARIES = {"bare": (1, False), "sc5": (5, False), "sc15": (15, False), "sc_tools": (5, True)}
TOL = 0.005

import urllib.request
def post(body, timeout=300):
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

SYS_PLAIN = ("You are an expert at probabilistic reasoning over causal Bayesian networks. Given a network over "
             "binary variables, compute P(Y=1|do(X=1)) using the truncated factorization. Reason step by step. "
             "End with: FINAL: <number>")
SYS_TOOLS = SYS_PLAIN + (" You may write ONE ```python``` block (numpy/itertools, sandboxed) to compute it; its "
                         "stdout is returned; then give FINAL: <number>.")
NUM = re.compile(r"FINAL:\s*([-+]?\d*\.?\d+)"); CODE = re.compile(r"```python\s*(.*?)```", re.S)

def run_code(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import numpy as np, itertools, math\n" + code); p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=5, env={"PATH": "/usr/bin"})
        return (r.stdout or "")[-400:] + (("\n[err]" + r.stderr[-120:]) if r.stderr else "")
    except Exception as e: return f"[exec {e}]"
    finally:
        try: os.unlink(p)
        except Exception: pass

def one(text, tools, rawdir, tag):
    sysmsg = SYS_TOOLS if tools else SYS_PLAIN
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": text}]
    tr = []
    for turn in range(4 if tools else 1):
        d = post({"model": "gpt-5.1", "messages": msgs, "max_completion_tokens": 8192})
        c = d["choices"][0]["message"]["content"] or ""; tr.append(c)
        m = NUM.search(c)
        if m:
            (rawdir / f"{tag}.txt").write_text("\n---\n".join(tr))
            try: return float(m.group(1))
            except Exception: return None
        if tools and CODE.search(c):
            msgs += [{"role": "assistant", "content": c}, {"role": "user", "content": f"stdout:\n{run_code(CODE.search(c).group(1))}\nGive FINAL: <number>."}]
        else:
            msgs += [{"role": "assistant", "content": c}, {"role": "user", "content": "Give FINAL: <number>."}]
    (rawdir / f"{tag}.txt").write_text("\n---\n".join(tr))
    return None

def adversary_answer(text, K, tools, rawdir, tag):
    vals = [v for k in range(K) if (v := one(text, tools, rawdir, f"{tag}_k{k}")) is not None]
    return (float(np.median(vals)) if vals else None), len(vals)

def gen_instances(size):
    rng = np.random.default_rng(BASE_SEED + size); out = []
    for i in range(N):
        d = scale.gen_dag(size, rng); g = scale.build_graph(d)
        truth = scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value
        out.append({"i": i, "size": size, "text": scale.render_text(d), "truth": round(float(truth), 6)})
    return out

def main():
    print(f"PREREG {PREREG_SHA[:12]}…  sizes={SIZES} N={N} adversaries={list(ADVERSARIES)}")
    rawdir = HERE / "raw"; rawdir.mkdir(exist_ok=True)
    cache = HERE / "rows.jsonl"; done = set()
    if cache.exists():
        for ln in cache.read_text().splitlines():
            r = json.loads(ln); done.add((r["size"], r["adv"], r["i"]))
    tasks = []
    for size in SIZES:
        for inst in gen_instances(size):
            for adv, (K, tools) in ADVERSARIES.items():
                if (size, adv, inst["i"]) not in done:
                    tasks.append((size, adv, K, tools, inst))
    print(f"  {len(tasks)} cells-instances to run (parallel)", flush=True)
    lock = threading.Lock(); t0 = time.time(); cnt = [0]
    def work(t):
        size, adv, K, tools, inst = t
        pred, nv = adversary_answer(inst["text"], K, tools, rawdir, f"s{size}_{adv}_i{inst['i']}")
        row = {"size": size, "adv": adv, "i": inst["i"], "truth": inst["truth"], "pred": pred, "nvalid": nv}
        with lock:
            with open(cache, "a") as f: f.write(json.dumps(row) + "\n")
            cnt[0] += 1
            if cnt[0] % 20 == 0: print(f"  [{cnt[0]}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in as_completed([ex.submit(work, t) for t in tasks]): pass
    print("done boundary scan.", flush=True)

if __name__ == "__main__":
    main()
