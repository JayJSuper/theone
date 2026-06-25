import importlib.util,sys,json,os,re,threading,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from math import comb
import numpy as np
spec=importlib.util.spec_from_file_location("scale",Path("experiments/baseline_b_v1_scale/run.py"))
scale=importlib.util.module_from_spec(spec); a=sys.argv; sys.argv=["x"]; spec.loader.exec_module(scale); sys.argv=a
KEY=os.environ["OPENAI_API_KEY"]; TOL=0.005; SIZE=16; N=60; SEED=20260624
def post(b):
    r=urllib.request.Request("https://api.openai.com/v1/chat/completions",data=json.dumps(b).encode(),
        headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(r,timeout=200) as x: return json.load(x)
NUM=re.compile(r"FINAL:\s*([-+]?\d*\.?\d+)")
def bare(text):
    d=post({"model":"gpt-5.1","messages":[
        {"role":"system","content":"Compute P(Y=1|do(X=1)) via truncated factorization. End with FINAL: <number>"},
        {"role":"user","content":text}],"max_completion_tokens":8192})
    c=d["choices"][0]["message"]["content"] or ""; m=NUM.search(c)
    return float(m.group(1)) if m else None
rng=np.random.default_rng(SEED+SIZE); insts=[]
for i in range(N):
    d=scale.gen_dag(SIZE,rng); g=scale.build_graph(d)
    t=scale.InterventionEngine(g).query_intervention(d["y"],1,{d["x"]:1}).value
    insts.append((i,scale.render_text(d),round(float(t),6)))
res={}; lock=threading.Lock()
def w(t):
    i,txt,truth=t; p=bare(txt)
    with lock: res[i]=(p,truth)
with ThreadPoolExecutor(max_workers=8) as ex:
    list(as_completed([ex.submit(w,t) for t in insts]))
Aok=[res[i][0] is not None and abs(res[i][0]-res[i][1])<=TOL for i in range(N)]
acc=sum(Aok)/N; ng=1.0-acc
b=sum(1 for o in Aok if not o); c=0  # C always right
p=min(1.0,2*sum(comb(b+c,k) for k in range(min(b,c)+1))/(2**(b+c))) if (b+c) else 1.0
print(json.dumps({"size":SIZE,"N":N,"bare_acc":round(acc,3),"net_gain_C_minus_bare":round(ng,3),
    "mcnemar_p":round(p,6),"C_right_bare_wrong":b},indent=2))
