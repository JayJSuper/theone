"""Consult peer models (DeepSeek, Gemini) on The One's open hard problems.

Different pretraining/posttraining -> different priors, openness, caution. We ask humbly for
their best unconventional ideas, then synthesize. "损有余而补不足" — borrow what we lack.
"""
import os, json, sys, urllib.request


def load(name):
    for l in open(os.path.expanduser("~/.theone_keys.env")):
        l = l.strip()
        if l.startswith("export "):
            l = l[7:]
        if l.startswith(name):
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


UA = "Mozilla/5.0 AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def ask_deepseek(prompt, system, model="deepseek-chat"):
    key = load("DEEPSEEK_API_KEY")
    body = {"model": model, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.7, "max_tokens": 4000}
    req = urllib.request.Request("https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": UA})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content", "") or "") + (m.get("content", "") or "") or json.dumps(r)[:500]


def ask_gemini(prompt, system, model="gemini-2.5-pro"):
    key = load("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3000}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        return r["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        return "Gemini HTTP %s: %s" % (e.code, e.read().decode()[:300])


SYSTEM = ("You are a sharp research collaborator with a DIFFERENT background than the asker. "
          "Give your most concrete, unconventional, specific ideas — name exact methods/papers, "
          "propose experiments. Disagree freely. Brevity over hedging.")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    prompt = open(sys.argv[2]).read() if len(sys.argv) > 2 else "Say hi."
    if which in ("deepseek", "all"):
        try:
            print("=" * 70 + "\nDEEPSEEK (deepseek-reasoner):\n" + "=" * 70)
            print(ask_deepseek(prompt, SYSTEM))
        except Exception as e:
            print("deepseek error:", type(e).__name__, str(e)[:300])
    if which in ("gemini", "all"):
        try:
            print("\n" + "=" * 70 + "\nGEMINI (2.0-flash):\n" + "=" * 70)
            print(ask_gemini(prompt, SYSTEM))
        except Exception as e:
            print("gemini error:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
