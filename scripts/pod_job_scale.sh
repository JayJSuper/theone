#!/usr/bin/env bash
# Heavy B-line scaling job for a fresh RunPod GPU. Clones the repo, installs, and runs the
# genuinely compute-bound native-core + B3 long-context scaling sweeps to large N/L. Self-logging.
# Usage on pod:  GHPAT=<pat> bash pod_job_scale.sh 2>&1 | tee /workspace/scale.log
set -e
cd /workspace
if [ ! -d theone ]; then
  git clone "https://${GHPAT}@github.com/JayJSuper/theone.git" theone
fi
cd theone
python -m pip install -q -U pip
python -m pip install -q numpy scipy pandas scikit-learn pgmpy 2>/dev/null || true
python -m pip install -q -e . 2>/dev/null || true
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "=========== LINE A: native causal core massive scaling ==========="
# continuous native engine reproducibility-stability vs N — push N to where it is genuinely heavy
THEONE_SCALE_NS="${SCALE_NS:-250000,1000000,4000000,16000000,64000000}" \
  python experiments/bline_scale/run.py || echo "bline_scale done/err"

echo "=========== LINE B: B1 real-scale latent causal (many seeds) ==========="
THEONE_B1_SEEDS="${B1_SEEDS:-0,1,2,3,4}" python experiments/bline_real_b1/run.py 2>/dev/null || echo "b1 done/err"

echo "=========== LINE C: B3 SSM extreme long-context scaling ==========="
THEONE_B3_LENGTHS="${B3_LENGTHS:-4096,16384,65536,262144}" \
  python experiments/bline_b3_longrange/run.py 2>/dev/null || echo "b3 done/err"

echo "=========== ALL SCALING LINES DONE ==========="
