"""PixelProof Studio: a zero-dependency local web UI to generate, inspect and stress-test tasks live."""
from __future__ import annotations

import base64
import json
import random
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .core import LEVELS, REGISTRY, png_bytes
from .models import make_model
from .pipeline import STAGES, build_pair
from .scoring import classify_error, parse

PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>PixelProof Studio</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#111;--ink2:#555;--line:#e3e1dc;--accent:#2a78d6;--ok:#0a7d32;--bad:#c62f2e;--chip:#eef3fb}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--panel:#1f1f1d;--ink:#f0efec;--ink2:#b9b8b0;--line:#34332f;--accent:#3987e5;--ok:#3fbf6a;--bad:#e66767;--chip:#22324a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
header b{font-size:18px;letter-spacing:-.01em}header span{color:var(--ink2);font-size:13px}
.ctl{display:flex;flex-wrap:wrap;gap:12px;align-items:end;padding:16px 20px}
label{font-size:12px;color:var(--ink2);display:flex;flex-direction:column;gap:4px}
select,input{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
button{font:inherit;font-weight:600;padding:8px 16px;border-radius:8px;border:0;background:var(--accent);color:#fff;cursor:pointer}
button.sec{background:transparent;color:var(--accent);border:1px solid var(--accent)}
main{padding:0 20px 40px;display:grid;gap:16px}
.q{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;font-size:17px}
.q small{display:block;color:var(--ink2);font-size:12.5px;margin-top:4px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:820px){.pair{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card img{width:100%;display:block;background:#fff}.card .b{padding:12px 14px}
.tag{display:inline-block;background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px;margin:2px 4px 2px 0}
.ans{font-size:20px;font-weight:700}.ok{color:var(--ok)}.bad{color:var(--bad)}
.gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.gate{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13px}
.gate b{display:block}.mono{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink2);white-space:pre-wrap}
</style></head><body>
<header><b>PixelProof Studio</b><span>verifiable visual task factory · counterfactual twins · independent pixel verifier</span>
<a id=replink href="/report" target=_blank style="margin-left:auto;color:var(--accent);font-weight:600;text-decoration:none;display:none">Open findings report ↗</a></header>
<div class=ctl>
<label>Family<select id=fam></select></label>
<label>Difficulty <span id=lvv>L2</span><input id=lv type=range min=0 max=4 value=2></label>
<label>Seed<input id=seed type=number value=1 style="width:110px"></label>
<button id=go>Generate pair</button><button class=sec id=rnd>Random seed</button>
<button class=sec id=ask style="display:none">Ask model</button>
</div>
<main>
<div class=q id=q>Pick a family and press Generate.</div>
<div class=gates id=gates></div>
<div class=pair id=pair></div>
</main>
<script>
const $=s=>document.querySelector(s);let cur=null,MODEL=null;
fetch('/api/families').then(r=>r.json()).then(d=>{MODEL=d.model;if(d.report)$('#replink').style.display='';for(const f of d.families){const o=document.createElement('option');o.value=f.name;o.textContent=f.title;$('#fam').appendChild(o)}
 if(MODEL){$('#ask').style.display='';$('#ask').textContent='Ask '+MODEL}gen()});
$('#lv').oninput=()=>$('#lvv').textContent='L'+$('#lv').value;
$('#go').onclick=gen;$('#rnd').onclick=()=>{$('#seed').value=Math.floor(Math.random()*1e6);gen()};
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function gen(){$('#q').textContent='Generating…';const p=new URLSearchParams({family:$('#fam').value,level:$('#lv').value,seed:$('#seed').value});
 const d=await (await fetch('/api/pair?'+p)).json();cur=d;if(d.error){$('#q').textContent=d.error;return}
 $('#q').innerHTML=esc(d.question)+'<small>Answer format: '+esc(d.format)+' · same question for both twins</small>';
 $('#gates').innerHTML=d.gates.map(g=>`<div class=gate><b class=ok>✓ ${esc(g[0])}</b>${esc(g[1])}</div>`).join('')+
  `<div class=gate><b>Funnel for this pair</b>${d.funnel.candidates} candidate(s) screened, ${d.funnel.rejected} rejected</div>`;
 $('#pair').innerHTML=['A','B'].map(r=>{const t=d[r];return `<div class=card><img src="${t.img}" alt="twin ${r}"><div class=b>
 <div>Twin ${r} · truth <span class="ans ok">${esc(t.answer)}</span>${t.tol?` <span class=mono>± ${t.tol}</span>`:''}</div>
 <div class=mono>independent verifier read: ${esc(t.verifier)}</div>
 <div>${Object.entries(t.knobs).map(([k,v])=>`<span class=tag>${esc(k)}: ${esc(v)}</span>`).join('')}</div>
 <div id=m${r}></div></div></div>`}).join('')}
$('#ask').onclick=async()=>{if(!cur)return;for(const r of ['A','B']){$('#m'+r).innerHTML='<span class=mono>asking…</span>'}
 const d=await (await fetch('/api/ask',{method:'POST',body:JSON.stringify({key:cur.key})})).json();
 for(const r of ['A','B']){const x=d[r];if(!x){continue}$('#m'+r).innerHTML=`<div style="margin-top:8px">${esc(MODEL)} says <span class="ans ${x.correct?'ok':'bad'}">${esc(x.pred)}</span> ${x.correct?'✓':'✗ ('+esc(x.kind)+')'}</div><div class=mono>${esc(x.raw.slice(-300))}</div>`}}
</script></body></html>"""

_STORE: dict[str, dict] = {}


def _pair(family: str, level: int, seed: int):
    from collections import Counter
    fam = REGISTRY[family]
    funnel = Counter()
    res = build_pair(fam, random.Random(seed * 7919 + level), level, None, funnel, [])
    if res is None:
        return {"error": "no valid pair found for this seed — try another"}
    pa, ia, pb, ib, checks = res
    q, fmt = fam.question(pa)
    key = f"{family}-{level}-{seed}"
    out = {"key": key, "question": q, "format": fmt, "gates": [(k, d) for k, d in STAGES],
           "funnel": {"candidates": funnel["candidates"], "rejected": funnel["candidates"] - 1}}
    store = {"q": q, "fmt": fmt}
    for role, p, img in (("A", pa, ia), ("B", pb, ib)):
        b = png_bytes(img)
        a = fam.answer(p)
        v = fam.verify(np.asarray(img), fam.public(p), p["q"])
        out[role] = {"img": "data:image/png;base64," + base64.b64encode(b).decode(), "answer": a.value, "tol": a.tol,
                     "verifier": round(v, 3) if isinstance(v, float) else v, "knobs": fam.knobs(p)}
        store[role] = {"png": b, "answer": a}
    _STORE[key] = store
    return out


def serve(port: int = 8765, model_spec: str | None = None, host: str = "127.0.0.1", report: str | None = None,
          open_browser: bool = False):
    from pathlib import Path
    model = make_model(model_spec) if model_spec else None
    report_path = Path(report) if report else None

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = dict(urllib.parse.parse_qsl(u.query))
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/report" and report_path and report_path.exists():
                return self._send(200, report_path.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/families":
                return self._send(200, json.dumps({"model": model.name if model else None, "levels": LEVELS,
                                                   "report": bool(report_path and report_path.exists()),
                                                   "families": [{"name": k, "title": f.title} for k, f in REGISTRY.items()]}))
            if u.path == "/api/pair":
                try:
                    d = _pair(qs.get("family", "gauge"), int(qs.get("level", 2)), int(qs.get("seed", 1)))
                except Exception as e:  # noqa: BLE001
                    d = {"error": str(e)}
                return self._send(200, json.dumps(d, default=str))
            self._send(404, "{}")

        def do_POST(self):
            if self.path != "/api/ask" or model is None:
                return self._send(404, "{}")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            st = _STORE.get(body.get("key"))
            if not st:
                return self._send(404, "{}")
            out = {}
            for role in ("A", "B"):
                try:
                    r = model.ask(st[role]["png"], st["q"], st["fmt"])
                    pred, _ = parse(r["text"], st[role]["answer"])
                    kind = classify_error(pred, st[role]["answer"])
                    out[role] = {"pred": pred, "correct": kind == "correct", "kind": kind, "raw": r["text"]}
                except Exception as e:  # noqa: BLE001
                    out[role] = {"pred": None, "correct": False, "kind": "error", "raw": str(e)}
            self._send(200, json.dumps(out, default=str))

    srv = None
    for p in range(port, port + 30):  # if the port is taken (e.g. another dev server), use the next free one
        try:
            srv = ThreadingHTTPServer((host, p), H)
            break
        except OSError:
            print(f"Port {p} is busy, trying {p + 1}...")
    if srv is None:
        raise SystemExit(f"No free port between {port} and {port + 29}; pass --port")
    port = srv.server_address[1]
    url = f"http://{host}:{port}"
    print(f"PixelProof Studio on {url}  (model: {model.name if model else 'none'})")
    if report_path and report_path.exists():
        print(f"Findings report on {url}/report")
    print("Press Ctrl+C to stop.")
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    srv.serve_forever()
