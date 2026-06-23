import importlib.util, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine
HERE=Path(__file__).parent
s=importlib.util.spec_from_file_location("tc",HERE/"run.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def ask(text,tmo=75):
    try:
        req=urllib.request.Request("https://api.openai.com/v1/chat/completions",
          data=json.dumps({"model":"gpt-5.1","messages":[{"role":"system","content":m.R.SYS},{"role":"user","content":text+m.R.PROTO}],"max_completion_tokens":4096}).encode(),
          headers={"Content-Type":"application/json","Authorization":f"Bearer {os.environ['OPENAI_API_KEY']}"},method="POST")
        with urllib.request.urlopen(req,timeout=tmo) as r: c=json.loads(r.read().decode())["choices"][0]["message"]["content"]
    except Exception: return None
    mm=list(re.finditer(r"ANSWER:\s*([0-9]*\.?[0-9]+)",c or "",re.I)); return float(mm[-1].group(1)) if mm else None
jf=open(HERE/"decisive2.jsonl","a")
done=set()
try:
    for l in open(HERE/"decisive2.jsonl"):
        if l.strip(): r=json.loads(l); done.add((r["label"],r["i"]))
except FileNotFoundError: pass
for label,k,d in [("k2_d45_LONGLOW",2,45),("k5_d0_SHORTHIGH",5,0),("k2_d0_ref",2,0)]:
    for i in range(12):
        if (label,i) in done: continue
        g=m.k_graph_padded(k,d,41000+1000*k+7*d+i)
        truth=round(InterventionEngine(g).query_intervention("Y",1,{"X":1}).value,6)
        text=m.R.render(g,k); w=len(text.split()); p=ask(text)
        jf.write(json.dumps({"label":label,"k":k,"d":d,"i":i,"truth":truth,"words":w,"pred":p})+"\n"); jf.flush()
jf.close()
open(HERE/"decisive2.DONE","w").write("done")
