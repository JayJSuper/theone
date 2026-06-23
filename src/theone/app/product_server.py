"""Unified A-line product web app — one entry, scenario-aware, sovereign history.

Run:  source ~/.theone_keys.env && .venv/bin/python -m theone.app.product_server
Open: http://localhost:8000
"""
from theone.app.product import TheOneProduct

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


PAGE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The One · 可信助手</title>
<style>
:root{--ink:#23201b;--soft:#5f574a;--faint:#9a9183;--line:#e8e3d8;--bg:#f7f5ef;--panel:#fff;
--green:#3f7a4e;--amber:#b06a1e;--red:#b23a2e;--blue:#3a6480;--grey:#7a7468;
--sans:-apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.7;font-weight:350}
.wrap{max-width:620px;margin:0 auto;padding:0 18px;min-height:100vh;display:flex;flex-direction:column}
header{padding:28px 2px 12px}
header h1{font-size:23px;font-weight:600}
header p{font-size:14px;color:var(--soft);margin-top:7px}
header .diff{font-size:13px;color:var(--amber);margin-top:9px;background:#fdf6ea;border-radius:8px;padding:8px 12px}
.bar{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--faint);padding:6px 2px}
.bar a{color:var(--blue);cursor:pointer;text-decoration:underline}
#log{flex:1;padding:6px 0;display:flex;flex-direction:column;gap:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:15px 17px}
.card.you{background:#23201b;color:#f3efe6;align-self:flex-end;max-width:85%;border:none;border-bottom-right-radius:5px}
.tag{display:inline-block;font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;margin-bottom:7px}
.t-health{background:#e9f2eb;color:var(--green)} .t-finance{background:#e9f0f5;color:var(--blue)}
.t-memory{background:#f3eee2;color:var(--amber)} .t-chat{background:#f1efe9;color:var(--grey)}
.badge{display:inline-block;font-size:12.5px;font-weight:700;padding:2px 10px;border-radius:20px;margin:0 0 7px 6px}
.b-danger{background:#fbeae7;color:var(--red)} .b-ok{background:#e9f2eb;color:var(--green)}
.b-abstain{background:#e9f0f5;color:var(--blue)} .b-ask,.b-chat{background:#f1ede3;color:var(--grey)}
.head{font-size:16px;font-weight:600;line-height:1.45}
.detail{font-size:14.5px;color:var(--soft);margin-top:7px;white-space:pre-wrap}
.basis{font-size:12.5px;color:var(--faint);margin-top:7px;font-family:ui-monospace,Menlo,monospace}
.action{font-size:14px;margin-top:9px;font-weight:500}
.guard{margin-top:11px;border-top:1px dashed var(--line);padding-top:10px;font-size:13.5px}
.guard.hit{background:#fbeae7;color:var(--red);border:none;border-radius:9px;padding:10px 12px;font-weight:500}
.guard .ai{color:var(--faint);font-size:12.5px}
.disc{font-size:11.5px;color:var(--faint);margin-top:12px;border-top:1px solid var(--line);padding-top:7px}
.ex{display:flex;gap:7px;flex-wrap:wrap;padding:4px 2px 0}
.ex span{font-size:12.5px;color:var(--blue);border:1px solid var(--line);border-radius:20px;padding:4px 11px;cursor:pointer;background:#fff}
form{display:flex;gap:8px;padding:13px 2px 20px;position:sticky;bottom:0;background:var(--bg)}
input{flex:1;padding:12px 15px;border:1px solid var(--line);border-radius:11px;font-size:15px;background:#fff;outline:none}
input:focus{border-color:var(--blue)}
button{padding:0 19px;border:none;border-radius:11px;background:var(--blue);color:#fff;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}
.dim{color:var(--faint)}
</style></head><body>
<div class="wrap">
<header>
  <h1>🛡 The One · 可信助手</h1>
  <p>用药安全、财务计算、记忆——能核验的我给你<b>可复算的依据</b>,核验不了的我<b>老实说不知道</b>。</p>
  <div class="diff">和普通 AI 最大的不同:<b>我不会为了显得有用就瞎编</b>;而且我会顺手验一下普通 AI 这次有没有说错。</div>
</header>
<div class="bar"><span id="status">连接中…</span> · <a onclick="hist()">查看我的记忆</a> · <a onclick="clr()">清空历史</a></div>
<div id="log"></div>
<div class="ex">
  <span onclick="ex(this)">华法林和阿司匹林能一起吃吗?</span>
  <span onclick="ex(this)">100万房贷利率4.9%30年月供多少?</span>
  <span onclick="ex(this)">10万年化6%复利20年到期多少?</span>
  <span onclick="ex(this)">阿莫西林和美托洛尔</span>
  <span onclick="ex(this)">记住下周三复诊</span>
</div>
<form id="f"><input id="t" autocomplete="off" placeholder="问用药安全、算房贷/复利、或让我记点事…"><button id="b" type="submit">发送</button></form>
</div>
<script>
const log=document.getElementById('log'),inp=document.getElementById('t'),btn=document.getElementById('b');
fetch('/health').then(r=>r.json()).then(d=>{document.getElementById('status').textContent=
  `挂载 AI: ${d.provider} · ${d.live?'在线':'离线'} · 场景: 用药/财务/记忆`;});
function ex(e){inp.value=e.textContent;inp.focus();}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function you(t){const c=document.createElement('div');c.className='card you';c.textContent=t;log.appendChild(c);sc();}
function sc(){window.scrollTo(0,document.body.scrollHeight);}
const SC={health:['t-health','用药核验'],finance:['t-finance','财务精算'],memory:['t-memory','主权记忆'],chat:['t-chat','普通对话']};
const BD={danger:'b-danger',ok:'b-ok',abstain:'b-abstain',ask:'b-ask',chat:'b-chat'};
function ans(r){
  const c=document.createElement('div');c.className='card';
  const sc=SC[r.scenario]||['t-chat','—'];
  let h=`<span class="tag ${sc[0]}">${sc[1]}</span>`;
  if(r.badge&&r.badge!=='chat')h+=`<span class="badge ${BD[r.badge]||'b-ask'}">${r.badge==='danger'?'有冲突':r.badge==='ok'?'已核验':r.badge==='abstain'?'我没把握':'请补充'}</span>`;
  h+=`<div class="head">${esc(r.headline||'')}</div>`;
  if(r.detail)h+=`<div class="detail">${esc(r.detail)}</div>`;
  if(r.basis)h+=`<div class="basis">${esc(r.basis)}</div>`;
  if(r.action)h+=`<div class="action">👉 ${esc(r.action)}</div>`;
  if(r.provenance)h+=`<div class="basis">${esc(r.provenance)}</div>`;
  if(r.guard){const hit=r.guard.indexOf('🚩')>=0;h+=`<div class="guard ${hit?'hit':''}">`;
    if(r.llm_said)h+=`<div class="ai">普通 AI 刚才说:"${esc(r.llm_said)}"</div>`;
    h+=`<div>${esc(r.guard)}</div></div>`;}
  if(r.disclaimer)h+=`<div class="disc">${esc(r.disclaimer)}</div>`;
  c.innerHTML=h;log.appendChild(c);sc();
}
async function go(){const t=inp.value.trim();if(!t)return;you(t);inp.value='';btn.disabled=true;
  const w=document.createElement('div');w.className='card dim';w.textContent='处理中…';log.appendChild(w);sc();
  try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
    const d=await r.json();w.remove();ans(d);}catch(e){w.textContent='出错: '+e;}
  btn.disabled=false;inp.focus();}
async function hist(){const d=await (await fetch('/history')).json();
  ans({scenario:'memory',badge:'ok',headline:`📒 你的历史(${d.items.length} 条,完全属于你)`,
    detail:d.items.map(x=>`· [${x.scenario}] ${x.q}`).join('\n')||'(暂无)'});}
async function clr(){await fetch('/clear',{method:'POST'});ans({scenario:'memory',badge:'ok',headline:'🗑 历史已彻底清空(数据归你,删了就没了)'});}
document.getElementById('f').addEventListener('submit',function(e){e.preventDefault();go();});
</script></body></html>"""


def create_server():
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed — pip install fastapi uvicorn")
    app = FastAPI(title="The One · 可信助手")
    prod = TheOneProduct(provider="deepseek")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/health")
    def health():
        return {"status": "ok", "provider": prod.llm.provider, "live": prod.llm.available(),
                "scenarios": list(prod.SCENARIOS)}

    @app.post("/ask")
    async def ask(req: Request):
        try:
            body = await req.json()
            return JSONResponse(prod.ask((body or {}).get("text", "")))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"scenario": "chat", "badge": "ask",
                                 "headline": "出错了,请重试", "detail": f"{type(e).__name__}"})

    @app.get("/history")
    def history():
        return {"items": prod.export_history()}

    @app.post("/clear")
    def clear():
        return {"cleared": prod.clear_history()}
    return app


def main(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(create_server(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
