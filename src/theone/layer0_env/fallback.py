"""L0 · device-selection + graceful CPU fallback policy.

Given detected hardware, choose a device and a policy. The contract: The One must RUN on
whatever is present — a GPU server, or a CPU-only laptop — degrading gracefully rather
than failing. The verifiable-kernel A-line is pure-CPU-capable (no GPU required); the
B-line native engine benefits from GPU but must still fall back to CPU for development.
"""
from __future__ import annotations
from dataclasses import dataclass

from theone.layer0_env.detect import HardwareInfo, detect_hardware


@dataclass
class DevicePolicy:
    device: str            # "cuda" | "cpu"
    reason: str
    use_mamba: bool        # prefer mamba-ssm kernel only on GPU; else self-impl SSM
    max_workers: int


def choose_policy(hw: HardwareInfo | None = None) -> DevicePolicy:
    hw = hw or detect_hardware()
    workers = max(1, (hw.cpu_cores or 2) - 1)
    if hw.has_gpu:
        return DevicePolicy(
            device="cuda", reason=f"GPU present: {hw.gpus[0].name}",
            use_mamba=True, max_workers=workers)
    return DevicePolicy(
        device="cpu",
        reason="no NVIDIA GPU detected — CPU fallback (kernel A-line is CPU-capable)",
        use_mamba=False, max_workers=workers)


__all__ = ["DevicePolicy", "choose_policy"]
