"""Abstract base class for inference runtime adapters.

Every backend (llama.cpp, vLLM, custom CUDA, etc.) implements this
interface so the planner and CLI are backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adhie.types import InferenceMetrics, RunPlan


@dataclass
class GenerateRequest:
    """A single inference request."""

    prompt: str = ""
    messages: list[dict[str, str]] | None = None  # Chat format
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    stream: bool = True
    stop: list[str] | None = None


@dataclass
class GenerateResponse:
    """Result of a generation request."""

    text: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"  # "stop", "length", "error"
    metrics: InferenceMetrics | None = None


class RuntimeAdapter(ABC):
    """Interface for inference runtime backends."""

    @abstractmethod
    def load(self, plan: RunPlan) -> None:
        """Load the model with the given execution plan.

        This may start a subprocess, initialize a library, etc.
        Should block until the backend is ready to serve requests.
        """
        ...

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Run inference and return the response with timing metrics."""
        ...

    @abstractmethod
    def generate_simple(self, prompt: str, max_tokens: int = 256) -> str:
        """Convenience method: send a prompt, return the text."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is alive and ready."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Stop the backend and release resources."""
        ...

    @abstractmethod
    def last_metrics(self) -> InferenceMetrics | None:
        """Return metrics from the most recent generate() call."""
        ...

