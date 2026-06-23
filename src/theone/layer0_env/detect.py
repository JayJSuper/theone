"""L0 · hardware/environment detection (pure stdlib, no torch/psutil dependency).

Detects GPU (via nvidia-smi if present), CPU cores, RAM, and free disk — degrading
gracefully on any platform (a macOS dev box with no NVIDIA GPU returns gpu=None, not an
error). Honest scope: this reports what is *actually* present, so deployment decisions
(see fallback.py) are made on real hardware, not assumptions. No exotic hardware
(photonic/neuromorphic) is referenced — only what can be bought and detected today.
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class GPUInfo:
    name: str
    memory_mb: Optional[int] = None


@dataclass
class HardwareInfo:
    platform: str
    cpu_cores: Optional[int]
    ram_gb: Optional[float]
    disk_free_gb: Optional[float]
    gpus: List[GPUInfo]

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["has_gpu"] = self.has_gpu
        return d


def _detect_ram_gb() -> Optional[float]:
    # works on Linux/macOS via sysconf; returns None where unavailable
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 2)
    except (ValueError, OSError, AttributeError):
        return None


def _detect_gpus() -> List[GPUInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            mem = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            gpus.append(GPUInfo(name=parts[0], memory_mb=mem))
        return gpus
    except (subprocess.SubprocessError, OSError, ValueError):
        return []


def detect_hardware() -> HardwareInfo:
    try:
        disk_free = round(shutil.disk_usage("/").free / 1e9, 1)
    except OSError:
        disk_free = None
    return HardwareInfo(
        platform=f"{platform.system()} {platform.machine()}",
        cpu_cores=os.cpu_count(),
        ram_gb=_detect_ram_gb(),
        disk_free_gb=disk_free,
        gpus=_detect_gpus(),
    )


__all__ = ["HardwareInfo", "GPUInfo", "detect_hardware"]
