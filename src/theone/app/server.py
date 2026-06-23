"""The One · web application — a local server with a chat UI over the A-line product.

Run:  source ~/.theone_keys.env && .venv/bin/python -m theone.app.server
Then open:  http://localhost:8000

Every answer shows its provenance: engine-verified causal results carry a recomputable
credential (and the mounted LLM's number is corroborated/refuted); generation is clearly
labelled UNVERIFIED. Sovereign memory is per-server-session.
"""
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.app import TheOneApp, CausalDomain

# FastAPI is optional; import at module level so route type-hints (Request) resolve via
# module globals. If absent, create_server() raises a clear error.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


def _recovery_domain():
    g = CausalGraph()
    for n in ("S", "T", "R"):
        g.add_variable(Variable(n))
    g.add_edge("S", "T"); g.add_edge("S", "R"); g.add_edge("T", "R")
    g.set_cpt("S", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("T", {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.3, 1: 0.7}})
    oR = list(g.parent_order("R"))
    vals = {(0, 0): .80, (0, 1): .90, (1, 0): .30, (1, 1): .55}
    g.set_cpt("R", {tuple(s if p == "S" else t for p in oR): {1: v, 0: round(1 - v, 2)}
                    for (s, t), v in vals.items()})
    return CausalDomain("recovery", g, {
        "treatment": "T", "treat": "T", "治疗": "T", "治療": "T", "疗法": "T", "用药": "T", "吃药": "T",
        "recovery": "R", "recover": "R", "康复": "R", "康復": "R", "痊愈": "R", "恢复": "R", "好转": "R",
        "severity": "S", "病情": "S", "严重": "S", "病重": "S",
        "__treatment__": "T", "__target__": "R"})


INDEX_HTML = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The One · 可验证认知内核</title>
<style>
:root{--ink:#1a1814;--soft:#5c5448;--faint:#938a7c;--line:#e7e2d7;--bg:#faf8f3;
--gold:#9a7b3f;--green:#46784e;--red:#a3543f;--panel:#fff;
--sans:-apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif;--mono:"SF Mono",Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-weight:350;line-height:1.6}
.wrap{max-width:760px;margin:0 auto;padding:0 18px;display:flex;flex-direction:column;height:100vh}
header{padding:20px 4px 14px;border-bottom:1px solid var(--line)}
header h1{font-size:20px;font-weight:600;letter-spacing:.02em}
header .sub{font-size:12px;color:var(--faint);margin-top:3px;font-family:var(--mono)}
#log{flex:1;overflow-y:auto;padding:18px 2px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:90%}
.msg.you{align-self:flex-end}
.bubble{padding:11px 15px;border-radius:14px;font-size:15px;white-space:pre-wrap;word-break:break-word}
.you .bubble{background:#1a1814;color:#f3efe6;border-bottom-right-radius:4px}
.one .bubble{background:var(--panel);border:1px solid var(--line);border-bottom-left-radius:4px}
.prov{font-size:11px;font-family:var(--mono);margin:5px 4px 0;letter-spacing:.02em}
.v-yes{color:var(--green)} .v-no{color:var(--gold)} .v-mem{color:var(--gold)}
.cred{margin-top:8px;border-top:1px dashed var(--line);padding-top:8px;font-size:12px;color:var(--soft);font-family:var(--mono)}
.cred .k{color:var(--faint)}
.guard{margin-top:6px;font-size:12.5px;padding:6px 10px;border-radius:8px}
.guard.refuted{background:#fbeeea;color:var(--red)}
.guard.corroborated{background:#eef4ee;color:var(--green)}
form{display:flex;gap:8px;padding:14px 2px 18px;border-top:1px solid var(--line)}
input{flex:1;padding:12px 14px;border:1px solid var(--line);border-radius:10px;font-size:15px;font-family:var(--sans);background:#fff;outline:none}
input:focus{border-color:var(--gold)}
button{padding:0 18px;border:none;border-radius:10px;background:var(--gold);color:#fff;font-size:14px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.ex{display:flex;gap:7px;flex-wrap:wrap;padding:6px 2px 0}
.ex span{font-size:11.5px;color:var(--gold);border:1px solid var(--line);border-radius:20px;padding:3px 10px;cursor:pointer;background:#fff}
.dim{color:var(--faint);font-size:12px}
</style></head><body>
<div class="wrap">
<header>
  <h1>太一 · The One <span class="dim">— 可验证认知内核</span></h1>
  <div class="sub" id="status">connecting…</div>
</header>
<div id="log"></div>
<div class="ex">
  <span onclick="ex(this)">治疗对康复的因果效应是多少?</span>
  <span onclick="ex(this)">what is the effect of treatment on recovery?</span>
  <span onclick="ex(this)">记住 Q3 发布延到 11 月</span>
  <span onclick="ex(this)">recall</span>
  <span onclick="ex(this)">写个 Python 平方函数</span>
</div>
<form id="f">
  <input id="t" autocomplete="off" placeholder="问点什么…(因果问题会被引擎验证)">
  <button id="b" type="submit">发送</button>
</form>
</div>
<script>
const log=document.getElementById('log'),inp=document.getElementById('t'),btn=document.getElementById('b');
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent=
    `挂载 LLM: ${d.provider} · ${d.live?'在线':'离线桩'} | 已注册因果域: ${d.domain} (T→R, 受 S 混杂)`;
});
function ex(e){inp.value=e.textContent;inp.focus();}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function addYou(t){const m=document.createElement('div');m.className='msg you';
  m.innerHTML=`<div class="bubble">${esc(t)}</div>`;log.appendChild(m);scroll();}
function addOne(r){
  const m=document.createElement('div');m.className='msg one';
  const v=r.verified?'<span class="v-yes">✓ 引擎已验证</span>':
    (r.track==='sovereign_memory'?'<span class="v-mem">✓ 主权记忆</span>':'<span class="v-no">○ 未验证(挂载 LLM)</span>');
  let cred='';
  if(r.e_value!==undefined&&r.e_value!==null){
    cred=`<div class="cred"><span class="k">凭证</span> · regime: ${esc(r.regime)}<br>`+
      `<span class="k">复算</span> ${r.recomputed_ok?'通过 ✓':'失败'} (gap ${Number(r.recompute_gap).toExponential(1)}) · `+
      `<span class="k">E-value</span> ${r.e_value}</div>`;
  }
  let guard='';
  if(r.verdict){guard=`<div class="guard ${r.verdict}">幻觉护栏:${esc(r.verdict_note)}</div>`;}
  let extra='';
  if(r.recent&&r.recent.length){extra='<div class="cred">'+r.recent.map(x=>'· '+esc(x)).join('<br>')+'</div>';}
  m.innerHTML=`<div class="bubble">${esc(r.answer)}${cred}${guard}${extra}</div><div class="prov">${v} · ${esc(r.provenance)}</div>`;
  log.appendChild(m);scroll();
}
function scroll(){log.scrollTop=log.scrollHeight;}
async function send(){
  const text=inp.value.trim();if(!text)return false;
  addYou(text);inp.value='';btn.disabled=true;
  const wait=document.createElement('div');wait.className='msg one';wait.innerHTML='<div class="bubble dim">思考中…</div>';
  log.appendChild(wait);scroll();
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const d=await r.json();wait.remove();addOne(d);
  }catch(e){wait.querySelector('.bubble').textContent='出错: '+e;}
  btn.disabled=false;inp.focus();return false;
}
document.getElementById('f').addEventListener('submit',function(e){e.preventDefault();send();});
</script></body></html>"""


def create_server():
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed — `pip install fastapi uvicorn`")
    app = FastAPI(title="The One · verifiable causal kernel")
    state = {"app": TheOneApp(provider="deepseek", domain=_recovery_domain())}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        a = state["app"]
        return {"status": "ok", "provider": a.llm.provider,
                "live": a.llm.available(), "domain": a.domain.name if a.domain else None}

    @app.post("/ask")
    async def ask(req: Request):
        try:
            body = await req.json()
            text = (body or {}).get("text", "")
            return JSONResponse(state["app"].ask(text))
        except Exception as e:  # noqa: BLE001 — never 500 the UI
            return JSONResponse({"track": "error", "verified": False,
                                 "provenance": "error", "answer": f"内部错误: {type(e).__name__}: {e}"})
    return app


def main(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(create_server(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
