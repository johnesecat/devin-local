"""Hardware probing for picking sensible defaults on first launch.

We don't add a hard ``psutil`` dependency: this module degrades gracefully
when probes fail (returning conservative defaults) so settings.py can call
into it on any platform without crashing.

The probes are intentionally cheap and read-only — no subprocess calls
that block on slow drivers (we only shell out for ``nvidia-smi`` with a
1 s timeout, and only when the binary is on ``PATH``).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """A snapshot of the local machine's compute resources."""

    total_ram_gb: float
    cpu_count: int
    has_nvidia_gpu: bool
    nvidia_gpu_vram_gb: float  # 0 if no GPU

    @property
    def has_gpu(self) -> bool:
        return self.has_nvidia_gpu and self.nvidia_gpu_vram_gb > 0

    @property
    def is_low_resource(self) -> bool:
        """True if this box can't comfortably run a 7-8 B model on CPU.

        Threshold: <16 GB RAM and no GPU. A 5 GB Q4 model needs ~6 GB free
        plus KV cache, and CPU eval at that size is multi-minute on most
        consumer hardware. Below 16 GB we recommend a 3 B model instead.
        """
        return (not self.has_gpu) and self.total_ram_gb < 16.0


def _read_meminfo_gb() -> float:
    """Linux ``/proc/meminfo`` MemTotal in GB. Returns 0 if unavailable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return float(parts[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _win_total_ram_gb() -> float:
    """Best-effort Windows total RAM via ``ctypes``. 0 on failure."""
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        if ok:
            return stat.ullTotalPhys / (1024**3)
    except (OSError, AttributeError, ImportError):
        pass
    return 0.0


def _macos_total_ram_gb() -> float:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=1.0, text=True
        ).strip()
        if out.isdigit():
            return int(out) / (1024**3)
    except (OSError, subprocess.SubprocessError):
        pass
    return 0.0


def _probe_total_ram_gb() -> float:
    # Fast paths per OS, no psutil dependency.
    val = _read_meminfo_gb()
    if val > 0:
        return val
    val = _win_total_ram_gb()
    if val > 0:
        return val
    val = _macos_total_ram_gb()
    if val > 0:
        return val
    return 0.0


def _probe_nvidia_vram_gb() -> float:
    """Return total VRAM across all NVIDIA GPUs, or 0 if none / unavailable."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0.0
    try:
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=1.0,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total_mb = 0.0
    for line in out.splitlines():
        line = line.strip()
        with contextlib.suppress(ValueError):
            total_mb += float(line)
    return total_mb / 1024.0


def probe() -> HardwareProfile:
    """Return a :class:`HardwareProfile` describing the local machine.

    Safe to call from any process. Each individual probe falls back to 0
    on failure, so a machine where every probe fails simply looks like a
    "low resource" CPU-only box.
    """
    ram_gb = _probe_total_ram_gb()
    cpu_count = os.cpu_count() or 1
    vram_gb = _probe_nvidia_vram_gb()
    return HardwareProfile(
        total_ram_gb=ram_gb,
        cpu_count=cpu_count,
        has_nvidia_gpu=vram_gb > 0,
        nvidia_gpu_vram_gb=vram_gb,
    )


# Largest-to-smallest. Each entry is ``(model_name, min_budget_gb)``: the
# minimum (RAM-derived or VRAM-derived) budget needed to host the model at
# Q4 with KV cache, OS overhead, and the Ollama runner accounted for.
# Tool-calling-capable everywhere.
_RECOMMENDATION_LADDER: tuple[tuple[str, float], ...] = (
    ("qwen2.5:14b", 14.0),  # ~9 GB weights + KV cache + runtime overhead
    ("llama3.1:8b", 6.0),  # ~5 GB weights + cache
    ("qwen2.5:7b", 6.0),
    ("llama3.2:3b", 3.0),  # ~2 GB weights + cache
    ("llama3.2:1b", 1.5),
)


def recommend_default_model(
    profile: HardwareProfile | None = None,
    *,
    installed: list[str] | None = None,
) -> str:
    """Pick the largest sensible default model for this machine.

    If ``installed`` is provided, the choice is constrained to models that
    are already on disk (so we don't suggest something the user hasn't
    pulled yet). If nothing on the ladder is installed, the first entry on
    the ladder whose hardware requirement is met is returned (caller can
    decide whether to ``ollama pull`` it).
    """
    if profile is None:
        profile = probe()

    if profile.has_gpu:
        budget_gb = profile.nvidia_gpu_vram_gb
        cpu_only = False
    else:
        # CPU class: be optimistic but leave headroom for the OS + ollama
        # runner. ~50% of system RAM is the usable inference budget once
        # you account for KV cache growth on multi-turn chats.
        budget_gb = max(1.5, profile.total_ram_gb * 0.5)
        cpu_only = True

    installed_set = set(installed or [])

    def _is_cpu_friendly(name: str) -> bool:
        """On CPU, never recommend a model bigger than 8 B even if RAM
        allows it — eval is multi-minute-per-token. Users who explicitly
        want a 14 B+ on CPU can pick it from the dropdown.
        """
        return not (cpu_only and name in {"qwen2.5:14b"})

    # Prefer an installed model that fits.
    if installed_set:
        for name, required_gb in _RECOMMENDATION_LADDER:
            if name in installed_set and required_gb <= budget_gb and _is_cpu_friendly(name):
                return name
        # Fall back to whatever is installed if nothing on the ladder
        # matches.
        ladder_names = {n for n, _ in _RECOMMENDATION_LADDER}
        for inst in sorted(installed_set):
            if inst not in ladder_names:
                return inst

    # Nothing installed yet (or installed list not provided): return the
    # largest ladder model the hardware can host.
    for name, required_gb in _RECOMMENDATION_LADDER:
        if required_gb <= budget_gb and _is_cpu_friendly(name):
            return name

    # Tiny machine that can't even fit a 1B model? Stick with the smallest.
    return _RECOMMENDATION_LADDER[-1][0]
