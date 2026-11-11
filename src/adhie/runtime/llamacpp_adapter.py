"""llama.cpp Runtime Adapter — manages a llama-server subprocess.

Translates a RunPlan into llama-server CLI arguments, starts the server
as a subprocess, communicates via its HTTP API, and collects timing metrics.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

from adhie.runtime.base import GenerateRequest, GenerateResponse, RuntimeAdapter
from adhie.telemetry.collector import TelemetryCollector
from adhie.types import InferenceMetrics, RunPlan


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787  # Avoid common ports
HEALTH_TIMEOUT = 120  # Max seconds to wait for server startup
HEALTH_POLL_INTERVAL = 1.0


class LlamaCppAdapter(RuntimeAdapter):
    """Wraps a llama-server subprocess as an ADHIE runtime.

    Lifecycle:
        adapter = LlamaCppAdapter("/path/to/llama-server")
        adapter.load(plan)          # starts subprocess, waits for ready
        response = adapter.generate(request)  # HTTP POST to /v1/chat/completions
        adapter.unload()            # kills subprocess
    """

    def __init__(
        self,
        server_path: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.server_path = server_path
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: subprocess.Popen | None = None
        self._last_metrics: InferenceMetrics | None = None
        self._current_plan: RunPlan | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, plan: RunPlan) -> None:
        """Start llama-server with the configuration from plan."""
        if self.process is not None:
            self.unload()

        self._current_plan = plan
        args = self._build_args(plan)

        # Log the command being run
        cmd_str = " ".join(args)
        print(f"[ADHIE] Starting llama-server...")
        print(f"[ADHIE] Command: {cmd_str}")

        # Start the server subprocess
        try:
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags if platform.system() == "Windows" else 0,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"llama-server not found at: {self.server_path}\n"
                "Run ./scripts/setup_backend.sh or pass --backend-path"
            )
        except PermissionError:
            raise PermissionError(
                f"Cannot execute llama-server at: {self.server_path}\n"
                "Check file permissions."
            )

        # Wait for the server to become ready
        if not self._wait_for_ready():
            # Collect error output
            stderr_output = ""
            if self.process and self.process.stderr:
                try:
                    stderr_output = self.process.stderr.read().decode(
                        "utf-8", errors="replace"
                    )[:2000]
                except Exception:
                    pass
            self.unload()
            raise RuntimeError(
                f"llama-server failed to start within {HEALTH_TIMEOUT}s.\n"
                f"stderr: {stderr_output}"
            )

        print(f"[ADHIE] Server ready at {self.base_url}")

    def unload(self) -> None:
        """Stop the server subprocess."""
        if self.process is None:
            return

        try:
            if platform.system() == "Windows":
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)

            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

        self.process = None
        self._current_plan = None
        print("[ADHIE] Server stopped.")

    def health_check(self) -> bool:
        """Check if the server is alive and ready."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Send a generation request and collect metrics."""
        if not self.health_check():
            raise RuntimeError("Server is not ready. Call load() first.")

        # Build request payload
        if request.messages:
            messages = request.messages
        else:
            messages = [{"role": "user", "content": request.prompt}]

        payload = {
            "model": "adhie-model",
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "stream": request.stream,
        }
        if request.stream:
            payload["stream_options"] = {"include_usage": True}
        if request.stop:
            payload["stop"] = request.stop

        collector = TelemetryCollector()
        pid = self.process.pid if self.process else None
        collector.start(prompt_tokens=0, pid=pid)

        if request.stream:
            response = self._generate_streaming(payload, collector)
        else:
            response = self._generate_non_streaming(payload, collector)

        response.metrics = collector.finalize()
        self._last_metrics = response.metrics

        # Try to get VRAM usage
        vram_bytes = self._get_vram_usage()
        if vram_bytes is not None and response.metrics:
            response.metrics.peak_vram_bytes = vram_bytes

        return response

    def generate_simple(self, prompt: str, max_tokens: int = 256) -> str:
        """Send a prompt and return just the text response."""
        request = GenerateRequest(prompt=prompt, max_tokens=max_tokens, stream=True)
        response = self.generate(request)
        return response.text

    def last_metrics(self) -> InferenceMetrics | None:
        """Return metrics from the most recent generate() call."""
        return self._last_metrics

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def _generate_streaming(
        self, payload: dict, collector: TelemetryCollector,
    ) -> GenerateResponse:
        """Handle SSE streaming response from /v1/chat/completions."""
        url = f"{self.base_url}/v1/chat/completions"
        full_text = ""
        token_count = 0
        finish_reason = "stop"
        prompt_tokens = 0

        try:
            with requests.post(url, json=payload, stream=True, timeout=300) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Check for usage object (sent in final chunk with choices=[])
                    usage = data.get("usage")
                    if usage:
                        pt = usage.get("prompt_tokens")
                        if pt is not None and pt > 0:
                            prompt_tokens = pt
                            collector._prompt_tokens = prompt_tokens

                    # Extract token content
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        full_text += content
                        token_count += 1

                        if token_count == 1:
                            collector.record_first_token()
                        else:
                            collector.record_token()

                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

        except requests.exceptions.Timeout:
            finish_reason = "error"
        except requests.exceptions.RequestException as e:
            finish_reason = "error"
            print(f"[ADHIE] Request error: {e}")

        if prompt_tokens > 0:
            collector._prompt_tokens = prompt_tokens

        return GenerateResponse(
            text=full_text,
            prompt_tokens=prompt_tokens,
            output_tokens=token_count,
            finish_reason=finish_reason,
        )

    def _generate_non_streaming(
        self, payload: dict, collector: TelemetryCollector,
    ) -> GenerateResponse:
        """Handle non-streaming response."""
        payload["stream"] = False
        url = f"{self.base_url}/v1/chat/completions"

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})

            output_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

            if output_tokens > 0:
                collector.record_first_token()
                collector.record_tokens(output_tokens - 1)
            collector._prompt_tokens = prompt_tokens

            return GenerateResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
            )
        except requests.exceptions.RequestException as e:
            return GenerateResponse(text="", finish_reason="error")

    # ------------------------------------------------------------------
    # CLI argument builder
    # ------------------------------------------------------------------

    def _build_args(self, plan: RunPlan) -> list[str]:
        """Translate a RunPlan into llama-server CLI arguments."""
        args = [
            self.server_path,
            "--model", plan.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(plan.context_length),
            "--threads", str(plan.cpu_threads),
            "--batch-size", str(plan.batch_size),
            "--ubatch-size", str(plan.ubatch_size),
            "--gpu-layers", str(plan.gpu_layers),
        ]

        # KV cache types
        if plan.kv_type_k and plan.kv_type_k != "f16":
            args.extend(["--cache-type-k", plan.kv_type_k])
        if plan.kv_type_v and plan.kv_type_v != "f16":
            args.extend(["--cache-type-v", plan.kv_type_v])

        # KV cache offload (in ADHIE, kv_offload=True means offload to RAM/CPU)
        if plan.kv_offload:
            args.append("--no-kv-offload")

        # Speculative decoding
        if plan.draft_model_path and plan.draft_device != "none" and plan.spec_type != "none":
            args.extend(["--model-draft", plan.draft_model_path])
            args.extend(["--spec-type", plan.spec_type])

            if plan.draft_device == "cpu":
                args.extend(["--device-draft", "none"])

            if plan.draft_threads and plan.draft_threads > 0:
                args.extend(["--threads-draft", str(plan.draft_threads)])

            if plan.draft_length > 0:
                args.extend(["--spec-draft-n-max", str(plan.draft_length)])

        # Flash attention if available (improves performance on modern GPUs)
        # Newer llama.cpp requires a value: on|off|auto
        args.extend(["--flash-attn", "auto"])

        return args

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_ready(self) -> bool:
        """Poll /health until the server responds 200 or timeout."""
        start = time.time()
        while time.time() - start < HEALTH_TIMEOUT:
            # Check if process died
            if self.process and self.process.poll() is not None:
                return False

            if self.health_check():
                return True

            time.sleep(HEALTH_POLL_INTERVAL)

        return False

    def _get_vram_usage(self) -> int | None:
        """Get current GPU VRAM usage via pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return mem.used
        except Exception:
            return None

    def get_server_log(self) -> str:
        """Read accumulated server stderr (non-blocking)."""
        if self.process and self.process.stderr:
            try:
                import select
                if hasattr(select, "select"):
                    # Unix: non-blocking read
                    import fcntl
                    fd = self.process.stderr.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    try:
                        return self.process.stderr.read().decode("utf-8", errors="replace")
                    except (BlockingIOError, TypeError):
                        return ""
                    finally:
                        fcntl.fcntl(fd, fcntl.F_SETFL, fl)
            except Exception:
                pass
        return ""


def find_llama_server() -> str:
    """Locate the llama-server binary.

    Search order:
      1. vendor/llama.cpp/build/bin/llama-server (project-local build)
      2. llama-server on PATH
      3. Common installation directories
    """
    # Check vendor directory (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    vendor_paths = [
        # Pre-built release extracted directly into vendor/llama.cpp/
        project_root / "vendor" / "llama.cpp" / "llama-server",
        project_root / "vendor" / "llama.cpp" / "llama-server.exe",
        # Source build in vendor/llama.cpp/build/bin/
        project_root / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server",
        project_root / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server.exe",
    ]
    for vp in vendor_paths:
        if vp.exists():
            return str(vp)

    # Check PATH
    path = shutil.which("llama-server")
    if path:
        return path

    # Windows: check common locations
    if platform.system() == "Windows":
        common_win = [
            Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-server.exe",
            Path("C:/llama.cpp/build/bin/Release/llama-server.exe"),
            Path("C:/llama.cpp/build/bin/llama-server.exe"),
            # Scoop / winget / chocolatey common locations
            Path.home() / "scoop" / "shims" / "llama-server.exe",
            Path("C:/ProgramData/chocolatey/bin/llama-server.exe"),
        ]
        # Also check the project vendor directory with Release subdir
        vendor_release = (
            project_root / "vendor" / "llama.cpp" / "build"
            / "bin" / "Release" / "llama-server.exe"
        )
        common_win.insert(0, vendor_release)
        # Also check vendor directory directly (pre-built release extraction)
        vendor_direct = project_root / "vendor" / "llama.cpp" / "llama-server.exe"
        common_win.insert(0, vendor_direct)

        for p in common_win:
            if p.exists():
                return str(p)

    raise FileNotFoundError(
        "llama-server not found. Either:\n"
        "  1. Download a pre-built release from:\n"
        "     https://github.com/ggml-org/llama.cpp/releases\n"
        "     and add llama-server to PATH, or\n"
        "  2. Build from source:\n"
        "     Linux/macOS: ./scripts/setup_backend.sh\n"
        "     Windows:     .\\scripts\\setup_backend.ps1\n"
        "  3. Pass --backend-path /path/to/llama-server"
    )

