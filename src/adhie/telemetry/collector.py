"""Telemetry Collector — capture timing metrics during inference.

Used by runtime adapters to measure TTFT, TPOT, ITL, tokens/s,
and resource utilization during a generation request.
"""

from __future__ import annotations

import statistics
import time

import psutil

from adhie.types import InferenceMetrics


class TelemetryCollector:
    """Collects timing metrics during a streaming generation request.

    Usage:
        collector = TelemetryCollector()
        collector.start(prompt_tokens=100)
        # ... receive first token ...
        collector.record_first_token()
        # ... receive each subsequent token ...
        collector.record_token()
        collector.record_token()
        # ... done ...
        metrics = collector.finalize()
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._first_token_time: float | None = None
        self._token_times: list[float] = []
        self._prompt_tokens: int = 0
        self._output_tokens: int = 0
        self._peak_ram_bytes: int = 0
        self._process: psutil.Process | None = None

    def start(self, prompt_tokens: int = 0, pid: int | None = None) -> None:
        """Begin timing a generation request."""
        self._start_time = time.perf_counter()
        self._first_token_time = None
        self._token_times = []
        self._prompt_tokens = prompt_tokens
        self._output_tokens = 0
        self._peak_ram_bytes = 0

        if pid:
            try:
                self._process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                self._process = None
        else:
            self._process = None

    def record_first_token(self) -> None:
        """Record when the first output token arrives."""
        now = time.perf_counter()
        self._first_token_time = now
        self._token_times.append(now)
        self._output_tokens = 1
        self._sample_resources()

    def record_token(self) -> None:
        """Record when a subsequent output token arrives."""
        now = time.perf_counter()
        self._token_times.append(now)
        self._output_tokens += 1

        # Sample resources periodically (every 10 tokens)
        if self._output_tokens % 10 == 0:
            self._sample_resources()

    def record_tokens(self, count: int) -> None:
        """Record multiple tokens arriving at once (batch/speculative)."""
        now = time.perf_counter()
        if self._first_token_time is None and count > 0:
            self._first_token_time = now
        self._token_times.append(now)
        self._output_tokens += count

    def finalize(self) -> InferenceMetrics:
        """Compute final metrics from recorded timestamps."""
        end_time = time.perf_counter()
        total_time = end_time - self._start_time
        self._sample_resources()

        metrics = InferenceMetrics(
            total_output_tokens=self._output_tokens,
            total_prompt_tokens=self._prompt_tokens,
            total_time_s=total_time,
            peak_ram_bytes=self._peak_ram_bytes,
        )

        # TTFT: time from start to first token
        if self._first_token_time is not None:
            metrics.ttft_ms = (self._first_token_time - self._start_time) * 1000

        # TPOT: average time per output token (excluding first)
        if self._output_tokens > 1 and self._first_token_time is not None:
            gen_time = end_time - self._first_token_time
            metrics.tpot_ms = (gen_time / (self._output_tokens - 1)) * 1000

        # ITL: inter-token latency (time between consecutive token events)
        if len(self._token_times) >= 2:
            itl_list = []
            for i in range(1, len(self._token_times)):
                itl = (self._token_times[i] - self._token_times[i - 1]) * 1000
                itl_list.append(itl)
            metrics.itl_ms_list = itl_list
            if itl_list:
                sorted_itl = sorted(itl_list)
                idx_95 = min(len(sorted_itl) - 1, int(len(sorted_itl) * 0.95))
                metrics.itl_p95_ms = sorted_itl[idx_95]

        # Tokens per second
        if total_time > 0:
            metrics.output_tokens_per_s = self._output_tokens / total_time

        # Prompt tokens per second
        if self._first_token_time is not None and self._prompt_tokens > 0:
            prompt_time = self._first_token_time - self._start_time
            if prompt_time > 0:
                metrics.prompt_tokens_per_s = self._prompt_tokens / prompt_time

        # CPU utilization (snapshot)
        try:
            metrics.cpu_utilization_pct = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        return metrics

    def _sample_resources(self) -> None:
        """Sample current resource usage."""
        if self._process:
            try:
                mem = self._process.memory_info()
                self._peak_ram_bytes = max(self._peak_ram_bytes, mem.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        else:
            try:
                self._peak_ram_bytes = max(
                    self._peak_ram_bytes,
                    psutil.virtual_memory().used,
                )
            except Exception:
                pass

