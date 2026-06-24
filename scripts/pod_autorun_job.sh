#!/usr/bin/env bash
# SSH-free cloud job — runs automatically on pod boot (as the container's dockerArgs/CMD), clones
# the PUBLIC repo, runs a real heavy job, and pushes results to a SECRET GitHub Gist so the
# (SSH-blocked) operator can retrieve them by listing gists. No inbound access to the pod needed.
#
# Requires env: GIST_TOKEN  (a GitHub token with ONLY the `gist` scope — minimal blast radius).
# Optional env: JOB ("scale" | "w2cg"), POD_TAG (label echoed into the result for matching).
set +e
TAG="${POD_TAG:-$RUNPOD_POD_ID}"
LOG=/workspace/run.log
mkdir -p /workspace; cd /workspace
{
  echo "=== theone autorun · tag=$TAG · $(uname -a) ==="
  python -c "import torch;print('CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" 2>&1
  rm -rf theone                                          # persisted volume may hold a stale clone
  git clone --depth 1 https://github.com/JayJSuper/theone.git theone 2>&1 | tail -2
  cd theone
  export PYTHONPATH="$PWD/src:$PYTHONPATH"                # make the `theone` package importable
  python -m pip install -q numpy scipy pandas scikit-learn pgmpy 2>&1 | tail -1

  case "${JOB:-scale}" in
    scale)
      echo "=== native causal core scaling (real compute-bound) ==="
      THEONE_SCALE_NS="${SCALE_NS:-1000000,4000000,16000000,64000000,256000000}" \
        python experiments/bline_scale/run.py 2>&1 ;;
    w2cg)
      echo "=== W2CG bert-large fine-tune at scale ==="
      python -m pip install -q transformers 2>&1 | tail -1
      cd experiments/bline_w2cg_transformer
      MODELS="${MODELS:-bert-large-uncased,roberta-large}" SEEDS="${SEEDS:-0,1,2}" EPOCHS="${EPOCHS:-10}" \
        python sweep_b200.py 2>&1 ;;
    realizer)
      echo "=== learned fluent realizer (T5) at scale ==="
      python -m pip install -q transformers sentencepiece 2>&1 | tail -1
      MODEL="${MODEL:-t5-base}" EPOCHS="${EPOCHS:-10}" \
        python experiments/bline_b2_learned_realizer/run.py 2>&1 ;;
    varstruct)
      echo "=== structure-general native do() at scale (B4 frontier) ==="
      NTR="${NTR:-150000}" NTE="${NTE:-8000}" WIDTH="${WIDTH:-256}" EPOCHS="${EPOCHS:-400}" \
        python experiments/bline_native_do_varstruct/run.py 2>&1 ;;
    gnn_batched)
      echo "=== REAL-SCALE batched size-invariant GNN do() (B4 ① real-scale) ==="
      NTR="${NTR:-200000}" WIDTH="${WIDTH:-64}" EPOCHS="${EPOCHS:-40}" \
        python experiments/bline_native_do_gnn_batched/run.py 2>&1 ;;
    realscale3)
      echo "=== 任务三: 真尺度 B1/B4 三种子重跑 + 指纹 (no deletion) ==="
      RD=/workspace/realscale3_results; mkdir -p "$RD"
      python -m pip install -q torch 2>&1 | tail -1 || true
      for S in 0 1 2; do
        echo "----- B4 gnn_batched SEED=$S -----"
        SEED=$S NTR="${NTR:-200000}" WIDTH=64 EPOCHS=40 RESULT_DIR="$RD" \
          python experiments/bline_native_do_gnn_batched/run.py 2>&1
        echo "----- B1 bline_scale SEED=$S -----"
        SEED=$S THEONE_SCALE_NS="${SCALE_NS:-1000000,16000000,256000000}" RESULT_DIR="$RD" \
          python experiments/bline_scale/run.py 2>&1
      done
      echo "===== RESULT JSON BUNDLE (for external audit) ====="
      for f in "$RD"/*.json; do echo "### $f"; cat "$f"; echo; sha256sum "$f"; done ;;
  esac
} > "$LOG" 2>&1

# push the log as a SECRET gist (operator lists gists with the same token to retrieve)
python - "$LOG" "$TAG" <<'PY'
import json, os, sys, urllib.request
log, tag = sys.argv[1], sys.argv[2]
body = json.dumps({"description": f"theone-result-{tag}", "public": False,
                   "files": {f"result-{tag}.log": {"content": open(log).read()[-90000:] or "(empty)"}}}).encode()
req = urllib.request.Request("https://api.github.com/gists", data=body, headers={
    "Authorization": "Bearer " + os.environ["GIST_TOKEN"],
    "Accept": "application/vnd.github+json", "User-Agent": "theone-pod"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    print("GIST_CREATED", r.get("id"), r.get("html_url"))
except Exception as e:
    print("GIST_PUSH_FAILED", str(e)[:200])
PY
sleep 30   # keep the container alive briefly so the push completes before exit
