"""RunPod orchestration for The One — provision a top GPU, run a job, tear it down.

Loads RUNPOD_API_KEY from ~/.theone_keys.env (never printed). Subcommands:
  gpus                      list GPU types + on-demand price (read-only)
  deploy <gpuTypeId>        create an on-demand pod (cuda12.8 image, sshd) -> prints pod id
  status <podId>            pod runtime + ssh endpoint
  ssh <podId>               print an ssh command (key at ~/.ssh/theone_runpod)
  kill <podId>              terminate the pod (ALWAYS do this when done)

Security: secret never echoed; tear pods down immediately after use; alert the user at ~$800.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"


def _key():
    env = Path.home() / ".theone_keys.env"
    for ln in env.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("export "):
            ln = ln[7:]
        if ln.startswith("RUNPOD_API_KEY="):
            return ln.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("RUNPOD_API_KEY not found in ~/.theone_keys.env")


# RunPod's GraphQL sits behind Cloudflare, which 403s (error 1010) a default urllib UA
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(f"{API}?api_key={_key()}", data=body, headers={
        "Content-Type": "application/json", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    if "errors" in out:
        raise SystemExit("GraphQL error: " + json.dumps(out["errors"])[:400])
    return out["data"]


def cmd_gpus():
    q = """query { gpuTypes {
        id displayName memoryInGb secureCloud communityCloud
        lowestPrice(input:{gpuCount:1}) { uninterruptablePrice minimumBidPrice }
    } }"""
    d = gql(q)
    rows = []
    for g in d["gpuTypes"]:
        lp = g.get("lowestPrice") or {}
        price = lp.get("uninterruptablePrice")
        rows.append((price if price else 9e9, g["id"], g["displayName"],
                     g.get("memoryInGb"), price, g.get("secureCloud"), g.get("communityCloud")))
    rows.sort(key=lambda r: -r[0] if r[0] < 9e9 else 0)        # priciest (most powerful) first
    print(f"{'gpuTypeId':<34} {'name':<26} {'GB':>4} {'$/hr':>7} {'secure':>6} {'comm':>5}")
    for _, gid, name, gb, price, sec, comm in rows:
        p = f"{price:.2f}" if price else "n/a"
        print(f"{gid:<34} {name[:26]:<26} {str(gb):>4} {p:>7} {str(sec):>6} {str(comm):>5}")


def cmd_deploy(gpu_type_id: str):
    pubkey = (Path.home() / ".ssh" / "theone_runpod.pub")
    pk = pubkey.read_text().strip() if pubkey.exists() else ""
    name = os.environ.get("POD_NAME", "theone-w2cg")
    # let the RunPod image's own startup wire SSH from PUBLIC_KEY (robust; no dockerArgs override)
    q = """mutation($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) { id imageName machineId }
    }"""
    variables = {"input": {
        "gpuTypeId": gpu_type_id, "cloudType": "SECURE", "gpuCount": 1,
        "volumeInGb": 60, "volumeMountPath": "/workspace", "containerDiskInGb": 60,
        "minVcpuCount": 8, "minMemoryInGb": 32,
        "name": name,
        # default: cuda12.9 for B200; override IMAGE=...cuda12.4... for fast-cached Hopper/Ampere boot
        "imageName": os.environ.get("IMAGE", "runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404"),
        "ports": "22/tcp",
        "env": [{"key": "PUBLIC_KEY", "value": pk}],
    }}
    d = gql(q, variables)
    print(json.dumps(d["podFindAndDeployOnDemand"], indent=2))


def _gist_api(method, path, token, data=None):
    req = urllib.request.Request("https://api.github.com" + path, data=data, method=method,
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                 "User-Agent": "theone"})
    return json.loads(urllib.request.urlopen(req, timeout=40).read())


def cmd_deploy_autorun(gpu_type_id: str):
    """SSH-free auto-run. Hosts scripts/pod_autorun_job.sh in a gist, then deploys a pod whose CMD
    curls+runs it: clone PUBLIC repo, run the job, push the result log to a SECRET gist. Retrieve
    with gist_retrieve.py. Env: GIST_TOKEN (required), JOB, SCALE_NS, IMAGE, POD_NAME."""
    token = os.environ.get("GIST_TOKEN", "")
    if not token:
        raise SystemExit("set GIST_TOKEN (gist-scope GitHub token)")
    job_sh = (Path(__file__).parent / "pod_autorun_job.sh").read_text()
    # host the bootstrap script in a gist (clean — no fragile inline escaping in dockerArgs)
    g = _gist_api("POST", "/gists", token, json.dumps({
        "description": "theone-bootstrap", "public": False,
        "files": {"pod_autorun_job.sh": {"content": job_sh}}}).encode())
    raw = next(iter(g["files"].values()))["raw_url"]
    job = os.environ.get("JOB", "scale"); scale_ns = os.environ.get("SCALE_NS", "1000000,4000000,16000000,64000000")
    boot = f"bash -lc 'curl -sL \"{raw}\" -o /tmp/j.sh && JOB={job} SCALE_NS={scale_ns} bash /tmp/j.sh'"
    q = """mutation($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) { id imageName machineId }
    }"""
    variables = {"input": {
        "gpuTypeId": gpu_type_id, "cloudType": "SECURE", "gpuCount": 1,
        "volumeInGb": 60, "volumeMountPath": "/workspace", "containerDiskInGb": 60, "minVcpuCount": 8, "minMemoryInGb": 32,
        "name": os.environ.get("POD_NAME", "theone-autorun"),
        "imageName": os.environ.get("IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"),
        "dockerArgs": boot,
        "env": [{"key": "GIST_TOKEN", "value": token}],
    }}
    d = gql(q, variables)
    print(json.dumps(d["podFindAndDeployOnDemand"], indent=2))


def cmd_status(pod_id: str):
    q = """query($id: String!) { pod(input:{podId:$id}) {
        id name desiredStatus runtime { uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type } }
    } }"""
    print(json.dumps(gql(q, {"id": pod_id})["pod"], indent=2))


def cmd_kill(pod_id: str):
    q = """mutation($id: String!) { podTerminate(input:{podId:$id}) }"""
    gql(q, {"id": pod_id})
    print(f"terminated {pod_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    c = sys.argv[1]
    if c == "gpus": cmd_gpus()
    elif c == "deploy": cmd_deploy(sys.argv[2])
    elif c == "deploy_autorun": cmd_deploy_autorun(sys.argv[2])
    elif c == "status": cmd_status(sys.argv[2])
    elif c == "kill": cmd_kill(sys.argv[2])
    else: raise SystemExit(__doc__)
