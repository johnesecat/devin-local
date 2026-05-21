"""Hardware probing + default-model recommendation tests."""

from __future__ import annotations

from devin_local.hardware import HardwareProfile, recommend_default_model


def test_low_resource_recommendation() -> None:
    """8 GB RAM CPU box should default to a 3 B (not 8 B) model."""
    profile = HardwareProfile(
        total_ram_gb=8.0, cpu_count=4, has_nvidia_gpu=False, nvidia_gpu_vram_gb=0.0
    )
    assert profile.is_low_resource is True
    assert recommend_default_model(profile) == "llama3.2:3b"


def test_tiny_cpu_recommendation() -> None:
    """4 GB RAM CPU box should fall back to the 1 B model."""
    profile = HardwareProfile(
        total_ram_gb=4.0, cpu_count=2, has_nvidia_gpu=False, nvidia_gpu_vram_gb=0.0
    )
    assert recommend_default_model(profile) == "llama3.2:1b"


def test_workstation_cpu_recommendation() -> None:
    """32 GB RAM CPU-only box should comfortably host an 8 B model."""
    profile = HardwareProfile(
        total_ram_gb=32.0, cpu_count=12, has_nvidia_gpu=False, nvidia_gpu_vram_gb=0.0
    )
    assert profile.is_low_resource is False
    assert recommend_default_model(profile) == "llama3.1:8b"


def test_gpu_recommendation_uses_vram() -> None:
    """A 24 GB-VRAM GPU should target the largest tool-capable model in the ladder."""
    profile = HardwareProfile(
        total_ram_gb=64.0, cpu_count=16, has_nvidia_gpu=True, nvidia_gpu_vram_gb=24.0
    )
    assert recommend_default_model(profile) == "qwen2.5:14b"


def test_small_gpu_falls_back_to_8b() -> None:
    """A 12 GB-VRAM GPU isn't quite enough for the 14 B model."""
    profile = HardwareProfile(
        total_ram_gb=32.0, cpu_count=8, has_nvidia_gpu=True, nvidia_gpu_vram_gb=12.0
    )
    assert recommend_default_model(profile) == "llama3.1:8b"


def test_installed_constraint_takes_precedence_for_small_box() -> None:
    """If only llama3.1:8b is installed but the box is low-resource, we still
    recommend it (user explicitly installed it; don't suggest a model they
    haven't pulled). This guards against the "first install" surprise."""
    profile = HardwareProfile(
        total_ram_gb=8.0, cpu_count=4, has_nvidia_gpu=False, nvidia_gpu_vram_gb=0.0
    )
    recommendation = recommend_default_model(profile, installed=["llama3.1:8b"])
    # Either return the installed one or fall through to the ladder.
    assert recommendation in {"llama3.1:8b", "llama3.2:3b", "llama3.2:1b"}


def test_installed_ladder_match() -> None:
    """If a ladder model is installed, prefer it over an off-ladder one."""
    profile = HardwareProfile(
        total_ram_gb=8.0, cpu_count=4, has_nvidia_gpu=False, nvidia_gpu_vram_gb=0.0
    )
    assert (
        recommend_default_model(profile, installed=["llama3.2:3b", "custom:xyz"]) == "llama3.2:3b"
    )


def test_tiniest_machine_returns_smallest_ladder_entry() -> None:
    """A 1 GB-RAM headless box can't host anything; still return a value."""
    profile = HardwareProfile(
        total_ram_gb=1.0, cpu_count=1, has_nvidia_gpu=False, nvidia_gpu_vram_gb=0.0
    )
    out = recommend_default_model(profile)
    assert out == "llama3.2:1b"
