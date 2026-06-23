"""Wait for each pod's proxy SSH to open, then immediately git-clone the repo and launch the
heavy bline_scale job in the background (nohup). No file transfer — clone only. Secrets (PAT)
read from ~/.theone_keys.env and never printed. Run in background; prints LAUNCHED per pod.
"""
import json, subprocess, time, urllib.request
from pathlib import Path

ENV = (Path.home() / ".theone_keys.env").read_text()
def secret(name):
    for ln in ENV.splitlines():
        ln = ln.strip().removeprefix("export ")
        if ln.startswith(name + "="):
            return ln.split("=", 1)[1].strip().strip('"').strip("'")
    return ""
RUNPOD = secret("RUNPOD_API_KEY")        # repo is PUBLIC -> clone needs no credential
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
KEY = str(Path.home() / ".ssh/theone_runpod")

# pod -> NS sweep (heavier on H200). bline_scale: reproducibility-stability vs N (native core).
PODS = {"e9y0ouvv4rt3gu": ("H200", "2000000,8000000,32000000,128000000"),
        "8vxx3783hlf1ro": ("H100", "1000000,4000000,16000000,64000000")}


def host(pid):
    q = 'query($id:String!){pod(input:{podId:$id}){machine{podHostId}}}'
    b = json.dumps({"query": q, "variables": {"id": pid}}).encode()
    r = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={RUNPOD}", data=b,
                               headers={"Content-Type": "application/json", "User-Agent": UA})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())["data"]["pod"]["machine"]["podHostId"]


def ssh(target, cmd, timeout=200):
    return subprocess.run(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no",
                           "-o", "ConnectTimeout=12", target, cmd],
                          capture_output=True, text=True, timeout=timeout)


def launch_cmd(ns):
    clone = "https://github.com/JayJSuper/theone.git"     # public; no credential on the wire
    return ("bash -lc '"
            "cd /workspace 2>/dev/null || cd /root; "
            "rm -rf theone; "
            f"git clone --depth 1 {clone} theone >/tmp/clone.log 2>&1 && cd theone && "
            "python -m pip install -q numpy scipy pandas scikit-learn pgmpy >/tmp/pip.log 2>&1; "
            f"nohup env THEONE_SCALE_NS={ns} python experiments/bline_scale/run.py "
            ">/workspace/scale.log 2>&1 & echo LAUNCHED_PID=$!'")


def main():
    launched = {}
    for it in range(40):                      # up to ~13 min
        for pid, (nm, ns) in PODS.items():
            if pid in launched:
                continue
            try:
                h = host(pid)
                probe = ssh(f"{h}@ssh.runpod.io", "echo UP", timeout=25)
                if probe.returncode == 0 and "UP" in probe.stdout:
                    print(f"[{nm}] ssh open -> launching bline_scale NS={ns}", flush=True)
                    r = ssh(f"{h}@ssh.runpod.io", launch_cmd(ns), timeout=300)
                    print(f"[{nm}] {r.stdout.strip()[-120:]} {('ERR:'+r.stderr.strip()[-120:]) if r.returncode else ''}", flush=True)
                    launched[pid] = nm
                else:
                    print(f"[{nm}] waiting ssh (~{it*20}s) {probe.stderr.strip()[:60]}", flush=True)
            except Exception as e:
                print(f"[{nm}] poll err {str(e)[:80]}", flush=True)
        if len(launched) == len(PODS):
            break
        time.sleep(20)
    print("LAUNCHED:", json.dumps(launched))


if __name__ == "__main__":
    main()
