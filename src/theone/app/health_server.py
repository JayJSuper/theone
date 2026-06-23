"""Plain-language web app for the med-interaction checker — the felt-value demo.

Run:  source ~/.theone_keys.env && .venv/bin/python -m theone.app.health_server
Open: http://localhost:8000
"""
from theone.app.health import HealthChecker, DISCLAIMER
from theone.layer1_perception.llm_client import LLMClient

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


PAGE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>用药安全核对 · The One</title>
<style>
:root{--ink:#23201b;--soft:#5f574a;--faint:#9a9183;--line:#e8e3d8;--bg:#f7f5ef;--panel:#fff;
--green:#3f7a4e;--amber:#b06a1e;--red:#b23a2e;--blue:#3a6480;
--sans:-apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.7;font-weight:350}
.wrap{max-width:600px;margin:0 auto;padding:0 18px;min-height:100vh;display:flex;flex-direction:column}
header{padding:32px 2px 16px}
header h1{font-size:24px;font-weight:600}
header p{font-size:14.5px;color:var(--soft);margin-top:8px}
header .diff{font-size:13.5px;color:var(--amber);margin-top:10px;background:#fdf6ea;border-radius:8px;padding:8px 12px}
#log{flex:1;padding:8px 0 8px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card.you{background:#23201b;color:#f3efe6;align-self:flex-end;max-width:85%;border:none;border-bottom-right-radius:5px}
.badge{display:inline-block;font-size:13px;font-weight:700;padding:3px 11px;border-radius:20px;margin-bottom:8px}
.b-danger{background:#fbeae7;color:var(--red)} .b-ok{background:#e9f2eb;color:var(--green)}
.b-abstain{background:#e9f0f5;color:var(--blue)} .b-ask{background:#f1ede3;color:var(--soft)}
.head{font-size:17px;font-weight:600;line-height:1.4}
.detail{font-size:15px;color:var(--soft);margin-top:8px}
.basis{font-size:13px;color:var(--faint);margin-top:8px;font-family:ui-monospace,Menlo,monospace}
.action{font-size:14.5px;margin-top:10px;font-weight:500}
.guard{margin-top:12px;border-top:1px dashed var(--line);padding-top:11px;font-size:14px}
.guard.hit{background:#fbeae7;color:var(--red);border:none;border-radius:9px;padding:11px 13px;font-weight:500}
.guard .ai{color:var(--faint);font-size:13px}
.disc{font-size:12px;color:var(--faint);margin-top:14px;border-top:1px solid var(--line);padding-top:8px}
.ex{display:flex;gap:8px;flex-wrap:wrap;padding:4px 2px 0}
.ex span{font-size:13px;color:var(--blue);border:1px solid var(--line);border-radius:20px;padding:5px 12px;cursor:pointer;background:#fff}
form{display:flex;gap:8px;padding:14px 2px 22px;position:sticky;bottom:0;background:var(--bg)}
input{flex:1;padding:13px 15px;border:1px solid var(--line);border-radius:11px;font-size:15px;background:#fff;outline:none}
input:focus{border-color:var(--blue)}
button{padding:0 20px;border:none;border-radius:11px;background:var(--blue);color:#fff;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}
.dim{color:var(--faint)}
</style></head><body>
<div class="wrap">
<header>
  <h1>💊 用药安全核对</h1>
  <p>把你在吃的两种药告诉我,我帮你核对有没有<b>已知的用药冲突</b>。</p>
  <div class="diff">和普通 AI 不同:没把握时我会<b>老实说"我不知道、请问药师"</b>,绝不瞎编。而且我会<b>顺手验一下普通 AI 这次有没有乱说</b>。</div>
</header>
<div id="log"></div>
<div class="ex">
  <span onclick="ex(this)">华法林和阿司匹林能一起吃吗?</span>
  <span onclick="ex(this)">伟哥和硝酸甘油</span>
  <span onclick="ex(this)">辛伐他汀和克拉霉素</span>
  <span onclick="ex(this)">二甲双胍和赖诺普利</span>
  <span onclick="ex(this)">阿莫西林和美托洛尔</span>
</div>
<form id="f"><input id="t" autocomplete="off" placeholder="例如:我同时吃华法林和阿司匹林,有问题吗?"><button id="b" type="submit">核对</button></form>
</div>
<script>
const log=document.getElementById('log'),inp=document.getElementById('t'),btn=document.getElementById('b');
function ex(e){inp.value=e.textContent;inp.focus();}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function you(t){const c=document.createElement('div');c.className='card you';c.textContent=t;log.appendChild(c);sc();}
function sc(){window.scrollTo(0,document.body.scrollHeight);}
const B={danger:['b-danger'],ok:['b-ok'],abstain:['b-abstain'],ask:['b-ask']};
function answer(r){
  const c=document.createElement('div');c.className='card';
  const b=(B[r.badge]||['b-ask'])[0];
  let h=`<span class="badge ${b}">${r.badge==='danger'?'已核实·有冲突':r.badge==='ok'?'已核实·无重大冲突':r.badge==='abstain'?'我没把握':'请补充'}</span>`;
  h+=`<div class="head">${esc(r.headline)}</div>`;
  if(r.detail)h+=`<div class="detail">${esc(r.detail)}</div>`;
  if(r.basis)h+=`<div class="basis">${esc(r.basis)}</div>`;
  if(r.action)h+=`<div class="action">👉 ${esc(r.action)}</div>`;
  if(r.guard){const hit=r.guard.indexOf('🚩')>=0;
    h+=`<div class="guard ${hit?'hit':''}">`;
    if(r.llm_said)h+=`<div class="ai">普通 AI 刚才说:"${esc(r.llm_said)}"</div>`;
    h+=`<div>${esc(r.guard)}</div></div>`;}
  if(r.disclaimer)h+=`<div class="disc">${esc(r.disclaimer)}</div>`;
  c.innerHTML=h;log.appendChild(c);sc();
}
async function go(){
  const t=inp.value.trim();if(!t)return;you(t);inp.value='';btn.disabled=true;
  const w=document.createElement('div');w.className='card dim';w.textContent='核对中…';log.appendChild(w);sc();
  try{const r=await fetch('/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
    const d=await r.json();w.remove();answer(d);}
  catch(e){w.textContent='出错: '+e;}
  btn.disabled=false;inp.focus();
}
document.getElementById('f').addEventListener('submit',function(e){e.preventDefault();go();});
</script></body></html>"""


def create_server():
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed — pip install fastapi uvicorn")
    app = FastAPI(title="The One · 用药安全核对")
    checker = HealthChecker(llm=LLMClient("deepseek"))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/health")
    def health():
        return {"status": "ok", "live": checker.llm.available(), "scenario": "drug-interactions"}

    @app.post("/check")
    async def check(req: Request):
        try:
            body = await req.json()
            return JSONResponse(checker.check((body or {}).get("text", "")))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"badge": "ask", "headline": "出错了,请重试",
                                 "detail": f"{type(e).__name__}", "disclaimer": DISCLAIMER})
    return app


def main(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(create_server(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
