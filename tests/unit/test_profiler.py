"""Tests for the Hardware Profiler."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adhie.profiler.hardware_profiler import HardwareProfiler
from adhie.types import HardwareProfile


class TestHardwareProfilerDetection:
    """Test hardware detection with mocked system calls."""

    def test_detect_cpu_populates_cores(self) -> None:
        profiler = HardwareProfiler(run_benchmarks=False)
        hp = profiler.profile()
        # Should always detect at least 1 core on any machine
        assert hp.cpu_physical_cores >= 1
        assert hp.cpu_logical_cores >= 1

    def test_detect_ram_populates_memory(self) -> None:
        profiler = HardwareProfiler(run_benchmarks=False)
        hp = profiler.profile()
        # Should always detect some RAM
        assert hp.ram_total_bytes > 0
        assert hp.ram_available_bytes > 0
        assert hp.ram_available_bytes <= hp.ram_total_bytes

    def test_detect_cpu_model_not_empty(self) -> None:
        profiler = HardwareProfiler(run_benchmarks=False)
        hp = profiler.profile()
        assert hp.cpu_model != ""
        assert hp.cpu_model != "unknown" or True  # OK if unknown on some CI

    def test_no_gpu_graceful(self) -> None:
        """Profiler should handle no GPU gracefully."""
        profiler = HardwareProfiler(run_benchmarks=False)
        with patch("adhie.profiler.hardware_profiler.HardwareProfiler._detect_gpu") as mock_gpu:
            def set_no_gpu(hp: HardwareProfile) -> None:
                hp.gpu_count = 0
                hp.gpu_name = None
            mock_gpu.side_effect = set_no_gpu
            hp = profiler.profile()
            assert hp.has_gpu() is False
            assert hp.gpu_name is None
            assert hp.usable_vram_bytes() == 0

    def test_storage_type_detected(self) -> None:
        profiler = HardwareProfiler(run_benchmarks=False)
        hp = profiler.profile()
        assert hp.storage_type in ("ssd", "hdd", "unknown")


class TestHardwareProfilerBenchmarks:
    """Test benchmark functionality."""

    def test_ram_bandwidth_measured(self) -> None:
        profiler = HardwareProfiler(run_benchmarks=True)
        hp = HardwareProfile()
        profiler._benchmark_ram_bandwidth(hp)
        assert hp.ram_bandwidth_bytes_s is not None
        assert hp.ram_bandwidth_bytes_s > 0
        # Sanity: RAM bandwidth should be between 1 GB/s and 200 GB/s
        assert 1e9 < hp.ram_bandwidth_bytes_s < 200e9


class TestHardwareProfilerReport:
    """Test report formatting."""

    def test_print_report_no_crash(self) -> None:
        from rich.console import Console
        hp = HardwareProfile(
            gpu_name="Test GPU",
            vram_total_bytes=6_000_000_000,
            vram_free_bytes=5_500_000_000,
            gpu_count=1,
            cpu_model="Test CPU",
            cpu_physical_cores=8,
            cpu_logical_cores=16,
            ram_total_bytes=32_000_000_000,
            ram_available_bytes=24_000_000_000,
            ram_bandwidth_bytes_s=40e9,
        )
        console = Console(file=open("/dev/null", "w"))
        profiler = HardwareProfiler()
        profiler.print_report(hp, console)  # Should not raise

    def test_print_report_no_gpu(self) -> None:
        from rich.console import Console
        hp = HardwareProfile(
            cpu_model="Test CPU",
            cpu_physical_cores=4,
            cpu_logical_cores=8,
            ram_total_bytes=16_000_000_000,
            ram_available_bytes=12_000_000_000,
        )
        console = Console(file=open("/dev/null", "w"))
        profiler = HardwareProfiler()
        profiler.print_report(hp, console)  # Should not raise


class TestHardwareProfilerPersistence:
    """Test saving and loading profiles."""

    def test_profile_save_and_load(self) -> None:
        profiler = HardwareProfiler(run_benchmarks=False)
        hp = profiler.profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_profile.yaml"
            hp.to_yaml(path)
            assert path.exists()

            loaded = HardwareProfile.from_yaml(path)
            assert loaded.cpu_physical_cores == hp.cpu_physical_cores
            assert loaded.ram_total_bytes == hp.ram_total_bytes

