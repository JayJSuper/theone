# B-line scaling — datacenter CUDA (NVIDIA H200) result

Cloud run on a RunPod NVIDIA H200 SXM (143 GB), continuous native path
(`cuda_standalone.py`, mirror of `NativeVerifiableEngine.estimate_continuous`):
TARNet ATE + split-half reproducibility-stability. Controllable continuous SCM, true ATE = 3.0.

| N (units) | native ATE | ATE error | repro-stability | wall-clock |
|----------:|-----------:|----------:|----------------:|-----------:|
| 100,000   | 3.0017     | 0.0017    | 0.9990          | 4.5 s      |
| 500,000   | 3.0103     | 0.0103    | 0.9980          | 16.7 s     |
| 1,000,000 | 3.0031     | 0.0031    | 0.9909          | 33.2 s     |
| 2,000,000 | 2.9939     | 0.0061    | 0.9952          | 66.2 s     |

Device: `cuda · NVIDIA H200`. 2,000,000 units (3 network trainings: full + 2 halves,
400 epochs, mini-batched) in 66 s.

## Read

At datacenter scale (up to 2M units) the continuous native engine holds the property that
NOTE-088 flagged as small-sample-noisy: reproducibility-stability stays ~0.99+ and ATE error
stays ~0.002–0.01. Combined with the Apple-Silicon-GPU sweep (N=1.5k→200k, stability
0.963→~1.0), the full curve shows reproducibility-stability is sample-hungry and saturates
near 1.0 with scale — the caveat is resolved, now confirmed on production CUDA hardware.

Honest scope: synthetic SCM with a known true ATE (controls the experiment); the metric is
split-half retrain agreement. Pod was torn down immediately after retrieval (no idle burn).
