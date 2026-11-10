"""Hardware Profiler — detect and benchmark local compute resources.

Collects static GPU/CPU/RAM info and runs optional microbenchmarks for
bandwidth measurements. All results are returned as a HardwareProfile
dataclass.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
from rich.console import Console
from rich.table import Table

from adhie.types import HardwareProfile


class HardwareProfiler:
    """Detect hardware capabilities and optionally run microbenchmarks."""

    def __init__(self, verbose: bool = False, run_benchmarks: bool = True) -> None:
        self.verbose = verbose
        self.run_benchmarks = run_benchmarks

    def profile(self) -> HardwareProfile:
        """Run full hardware profiling and return a HardwareProfile."""
        hp = HardwareProfile()

        # CPU info
        self._detect_cpu(hp)

        # System memory
        self._detect_ram(hp)

        # GPU info
        self._detect_gpu(hp)

        # Storage info
        self._detect_storage(hp)

        # Run bandwidth benchmarks if requested
        if self.run_benchmarks:
            self._benchmark_ram_bandwidth(hp)
            if hp.has_gpu():
                self._benchmark_gpu_bandwidth(hp)

        return hp

    # ------------------------------------------------------------------
    # CPU Detection
    # ------------------------------------------------------------------

    def _detect_cpu(self, hp: HardwareProfile) -> None:
        """Detect CPU model, core counts, frequency, and SIMD flags."""
        hp.cpu_physical_cores = psutil.cpu_count(logical=False) or 0
        hp.cpu_logical_cores = psutil.cpu_count(logical=True) or 0

        freq = psutil.cpu_freq()
        if freq:
            hp.cpu_freq_mhz = freq.max or freq.current

        system = platform.system()

        if system == "Linux":
            hp.cpu_model = self._read_cpu_model_linux()
            hp.simd_flags = self._read_cpu_flags_linux()
        elif system == "Darwin":
            hp.cpu_model = self._read_cpu_model_macos()
            hp.simd_flags = self._read_cpu_flags_macos()
        else:
            hp.cpu_model = platform.processor() or "unknown"
            hp.simd_flags = []

    def _read_cpu_model_linux(self) -> str:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"

    def _read_cpu_flags_linux(self) -> list[str]:
        simd_keywords = {"avx", "avx2", "avx512f", "avx512bw", "avx512vl",
                         "sse", "sse2", "sse3", "sse4_1", "sse4_2",
                         "amx_tile", "amx_int8", "amx_bf16", "f16c", "fma"}
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        all_flags = line.split(":", 1)[1].strip().split()
                        return sorted(f for f in all_flags if f in simd_keywords)
        except OSError:
            pass
        return []

    def _read_cpu_model_macos(self) -> str:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return platform.processor() or "unknown"

    def _read_cpu_flags_macos(self) -> list[str]:
        flags = []
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                features = result.stdout.strip().lower().split()
                simd_keywords = {"avx1.0", "avx2.0", "avx512f", "sse", "sse2",
                                 "sse3", "sse4.1", "sse4.2", "fma", "f16c"}
                flags = sorted(f for f in features if f in simd_keywords)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return flags

    # ------------------------------------------------------------------
    # RAM Detection
    # ------------------------------------------------------------------

    def _detect_ram(self, hp: HardwareProfile) -> None:
        mem = psutil.virtual_memory()
        hp.ram_total_bytes = mem.total
        hp.ram_available_bytes = mem.available

    # ------------------------------------------------------------------
    # GPU Detection (NVIDIA via pynvml)
    # ------------------------------------------------------------------

    def _detect_gpu(self, hp: HardwareProfile) -> None:
        """Detect NVIDIA GPU properties via pynvml."""
        try:
            import pynvml  # noqa: F811 — nvidia-ml-py provides this
            pynvml.nvmlInit()
        except Exception:
            # No NVIDIA GPU or driver not available
            hp.gpu_count = 0
            hp.gpu_name = None
            if self.verbose:
                print("  [GPU] No NVIDIA GPU detected or pynvml unavailable")
            return

        try:
            hp.gpu_count = pynvml.nvmlDeviceGetCount()
            if hp.gpu_count == 0:
                return

            # Use first GPU (primary device)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            hp.gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(hp.gpu_name, bytes):
                hp.gpu_name = hp.gpu_name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            hp.vram_total_bytes = mem_info.total
            hp.vram_free_bytes = mem_info.free

            # Compute capability
            try:
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                if isinstance(major, tuple):
                    hp.gpu_compute_capability = major
                else:
                    minor = 0  # Fallback
                    hp.gpu_compute_capability = (major, minor)
            except Exception:
                hp.gpu_compute_capability = None

            # PCIe info
            try:
                hp.pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                hp.pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            except Exception:
                pass

            # CUDA / driver version
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8")
                hp.driver_version = driver
            except Exception:
                pass

            try:
                cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                major = cuda_ver // 1000
                minor = (cuda_ver % 1000) // 10
                hp.cuda_version = f"{major}.{minor}"
            except Exception:
                pass

        except Exception as e:
            if self.verbose:
                print(f"  [GPU] Error during GPU detection: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Storage Detection
    # ------------------------------------------------------------------

    def _detect_storage(self, hp: HardwareProfile) -> None:
        """Detect storage type (basic heuristic)."""
        system = platform.system()
        if system == "Darwin":
            # macOS — most modern Macs use SSD
            hp.storage_type = "ssd"
        elif system == "Windows":
            try:
                # Use PowerShell to detect media type
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-PhysicalDisk | Select-Object MediaType | "
                     "Format-Table -HideTableHeaders"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    output = result.stdout.strip().lower()
                    if "ssd" in output or "solid" in output:
                        hp.storage_type = "ssd"
                    elif "hdd" in output or "unspecified" in output:
                        hp.storage_type = "hdd"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            # Fallback: check if system drive is on NVMe
            if hp.storage_type == "unknown":
                try:
                    result = subprocess.run(
                        ["powershell", "-Command",
                         "Get-PhysicalDisk | Select-Object BusType | "
                         "Format-Table -HideTableHeaders"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode == 0 and "nvme" in result.stdout.lower():
                        hp.storage_type = "ssd"
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
        elif system == "Linux":
            try:
                # Check if root device is rotational
                result = subprocess.run(
                    ["lsblk", "-d", "-o", "name,rota", "--noheadings"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        parts = line.split()
                        if len(parts) == 2:
                            if parts[1] == "0":
                                hp.storage_type = "ssd"
                            else:
                                hp.storage_type = "hdd"
                            break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        else:
            hp.storage_type = "unknown"

    # ------------------------------------------------------------------
    # Microbenchmarks
    # ------------------------------------------------------------------

    def _benchmark_ram_bandwidth(self, hp: HardwareProfile) -> None:
        """Estimate CPU memory bandwidth via large array copy."""
        try:
            size = 256 * 1024 * 1024  # 256 MB
            a = np.ones(size // 8, dtype=np.float64)  # 256 MB
            b = np.empty_like(a)

            # Warmup
            np.copyto(b, a)

            # Timed runs
            times = []
            for _ in range(5):
                start = time.perf_counter()
                np.copyto(b, a)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            median_time = sorted(times)[len(times) // 2]
            # Read + Write = 2 × size
            hp.ram_bandwidth_bytes_s = (2 * size) / median_time

            if self.verbose:
                print(f"  [RAM] Bandwidth: {hp.ram_bandwidth_bytes_s / 1e9:.1f} GB/s "
                      f"(median of 5 runs)")
        except Exception as e:
            if self.verbose:
                print(f"  [RAM] Bandwidth benchmark failed: {e}")

    def _benchmark_gpu_bandwidth(self, hp: HardwareProfile) -> None:
        """Measure H2D and D2H bandwidth using CuPy or PyCUDA."""
        # Try CuPy first (simpler API)
        if self._benchmark_gpu_bandwidth_cupy(hp):
            return
        # Fall back to nvidia-smi theoretical bandwidth
        self._estimate_gpu_bandwidth_from_spec(hp)

    def _benchmark_gpu_bandwidth_cupy(self, hp: HardwareProfile) -> bool:
        """Measure GPU bandwidth with CuPy. Returns True if successful."""
        try:
            import cupy as cp

            size_bytes = 64 * 1024 * 1024  # 64 MB
            host_arr = np.ones(size_bytes // 4, dtype=np.float32)

            # Warmup
            d_arr = cp.asarray(host_arr)
            _ = cp.asnumpy(d_arr)
            cp.cuda.stream.get_current_stream().synchronize()

            # H2D benchmark
            h2d_times = []
            for _ in range(5):
                start = time.perf_counter()
                d_arr = cp.asarray(host_arr)
                cp.cuda.stream.get_current_stream().synchronize()
                h2d_times.append(time.perf_counter() - start)

            # D2H benchmark
            d2h_times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = cp.asnumpy(d_arr)
                cp.cuda.stream.get_current_stream().synchronize()
                d2h_times.append(time.perf_counter() - start)

            hp.h2d_bandwidth_bytes_s = size_bytes / sorted(h2d_times)[2]
            hp.d2h_bandwidth_bytes_s = size_bytes / sorted(d2h_times)[2]

            if self.verbose:
                print(f"  [GPU] H2D: {hp.h2d_bandwidth_bytes_s / 1e9:.1f} GB/s, "
                      f"D2H: {hp.d2h_bandwidth_bytes_s / 1e9:.1f} GB/s")

            # GPU memory bandwidth (device-to-device copy)
            size_gpu = 128 * 1024 * 1024  # 128 MB
            a_gpu = cp.ones(size_gpu // 4, dtype=cp.float32)
            b_gpu = cp.empty_like(a_gpu)

            # Warmup
            cp.copyto(b_gpu, a_gpu)
            cp.cuda.stream.get_current_stream().synchronize()

            gpu_times = []
            for _ in range(5):
                start = time.perf_counter()
                cp.copyto(b_gpu, a_gpu)
                cp.cuda.stream.get_current_stream().synchronize()
                gpu_times.append(time.perf_counter() - start)

            hp.gpu_mem_bandwidth_bytes_s = (2 * size_gpu) / sorted(gpu_times)[2]

            if self.verbose:
                print(f"  [GPU] Memory BW: {hp.gpu_mem_bandwidth_bytes_s / 1e9:.1f} GB/s")

            return True
        except ImportError:
            if self.verbose:
                print("  [GPU] CuPy not available, skipping GPU bandwidth benchmark")
            return False
        except Exception as e:
            if self.verbose:
                print(f"  [GPU] CuPy benchmark failed: {e}")
            return False

    def _estimate_gpu_bandwidth_from_spec(self, hp: HardwareProfile) -> None:
        """Fallback: estimate PCIe bandwidth from reported link data."""
        if hp.pcie_gen and hp.pcie_width:
            # PCIe bandwidth per lane per direction (approximate GB/s)
            bw_per_lane = {3: 0.985e9, 4: 1.969e9, 5: 3.938e9}
            lane_bw = bw_per_lane.get(hp.pcie_gen, 1.0e9)
            theoretical = lane_bw * hp.pcie_width
            # Assume ~80% efficiency
            hp.h2d_bandwidth_bytes_s = theoretical * 0.8
            hp.d2h_bandwidth_bytes_s = theoretical * 0.8

            if self.verbose:
                print(f"  [GPU] PCIe estimated BW: "
                      f"{hp.h2d_bandwidth_bytes_s / 1e9:.1f} GB/s "
                      f"(Gen{hp.pcie_gen} x{hp.pcie_width}, 80% efficiency)")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, hp: HardwareProfile, console: Console) -> None:
        """Print a formatted hardware profile report."""
        table = Table(title="Hardware Profile", show_header=True,
                      header_style="bold cyan")
        table.add_column("Component", style="bold")
        table.add_column("Property")
        table.add_column("Value", justify="right")

        # GPU
        if hp.has_gpu():
            table.add_row("GPU", "Name", hp.gpu_name or "N/A")
            table.add_row("", "VRAM Total",
                          f"{hp.vram_total_bytes / 1e9:.1f} GB")
            table.add_row("", "VRAM Free",
                          f"{hp.vram_free_bytes / 1e9:.1f} GB")
            table.add_row("", "VRAM Usable (after margin)",
                          f"{hp.usable_vram_bytes() / 1e9:.1f} GB")
            if hp.gpu_compute_capability:
                table.add_row("", "Compute Capability",
                              f"{hp.gpu_compute_capability[0]}.{hp.gpu_compute_capability[1]}")
            if hp.pcie_gen:
                table.add_row("", "PCIe",
                              f"Gen{hp.pcie_gen} x{hp.pcie_width}")
            table.add_row("", "GPU Count", str(hp.gpu_count))
        else:
            table.add_row("GPU", "Status", "[yellow]No GPU detected[/yellow]")

        table.add_section()

        # CPU
        table.add_row("CPU", "Model", hp.cpu_model)
        table.add_row("", "Cores",
                      f"{hp.cpu_physical_cores}P / {hp.cpu_logical_cores}L")
        if hp.cpu_freq_mhz:
            table.add_row("", "Max Frequency", f"{hp.cpu_freq_mhz:.0f} MHz")
        if hp.simd_flags:
            table.add_row("", "SIMD", ", ".join(hp.simd_flags))

        table.add_section()

        # RAM
        table.add_row("RAM", "Total",
                      f"{hp.ram_total_bytes / 1e9:.1f} GB")
        table.add_row("", "Available",
                      f"{hp.ram_available_bytes / 1e9:.1f} GB")

        table.add_section()

        # Bandwidths
        if hp.ram_bandwidth_bytes_s:
            table.add_row("Bandwidth", "RAM",
                          f"{hp.ram_bandwidth_bytes_s / 1e9:.1f} GB/s")
        if hp.h2d_bandwidth_bytes_s:
            table.add_row("", "H2D (PCIe)",
                          f"{hp.h2d_bandwidth_bytes_s / 1e9:.1f} GB/s")
        if hp.d2h_bandwidth_bytes_s:
            table.add_row("", "D2H (PCIe)",
                          f"{hp.d2h_bandwidth_bytes_s / 1e9:.1f} GB/s")
        if hp.gpu_mem_bandwidth_bytes_s:
            table.add_row("", "GPU Memory",
                          f"{hp.gpu_mem_bandwidth_bytes_s / 1e9:.1f} GB/s")

        table.add_section()

        # Runtime
        table.add_row("Runtime", "Storage", hp.storage_type)
        if hp.cuda_version:
            table.add_row("", "CUDA", hp.cuda_version)
        if hp.driver_version:
            table.add_row("", "Driver", hp.driver_version)
        table.add_row("", "Safety Margin",
                      f"{hp.vram_safety_margin_bytes / 1e6:.0f} MB")
        table.add_row("", "Fingerprint", hp.fingerprint())

        console.print(table)

