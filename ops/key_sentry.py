"""Key sentry — local, zero-cost guardrail for the shared keys (Jay authorized
direct use; this watches for theft/runaway). Reads usage we can cheaply observe:
- DeepSeek: official balance endpoint (drop = spend).
Records a timestamped snapshot to ops/key_usage_log.jsonl. Run periodically;
flags a balance drop faster than our own known spend. Keys read from env, never
logged. NOT committed with any secret."""
import json, os, time, urllib.request
from pathlib import Path
LOG = Path(__file__).parent / "key_usage_log.jsonl"

def deepseek_balance():
    req = urllib.request.Request("https://api.deepseek.com/user/balance",
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return float(d["balance_infos"][0]["total_balance"])

def main():
    snap = {"unix": int(time.time()), "deepseek_cny": deepseek_balance()}
    prev = None
    if LOG.exists():
        lines = [l for l in LOG.read_text().splitlines() if l.strip()]
        if lines: prev = json.loads(lines[-1])
    LOG.open("a").write(json.dumps(snap) + "\n")
    if prev:
        drop = prev["deepseek_cny"] - snap["deepseek_cny"]
        dt_h = (snap["unix"] - prev["unix"]) / 3600
        rate = drop / dt_h if dt_h else 0
        flag = "⚠️ ALERT: spend >¥20/h" if rate > 20 else "ok"
        print(f"DeepSeek ¥{snap['deepseek_cny']:.2f} | Δ¥{drop:+.2f} over {dt_h:.1f}h "
              f"({rate:+.1f}/h) {flag}")
    else:
        print(f"DeepSeek ¥{snap['deepseek_cny']:.2f} (baseline snapshot)")

if __name__ == "__main__":
    main()
