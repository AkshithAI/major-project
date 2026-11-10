"""Profile Cache — persist and reuse tuned RunPlans.

Caches the mapping (hardware_fingerprint, model_fingerprint, workload_hash)
→ RunPlan + metrics, so that subsequent runs with the same setup skip the
autotuning phase entirely.

Cache entries are stored as YAML files in ``~/.adhie/profiles/``.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from adhie.types import (
    HardwareProfile,
    InferenceMetrics,
    ModelProfile,
    RunPlan,
    WorkloadConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".adhie" / "profiles"


class ProfileCache:
    """Cache (hardware, model, workload) → RunPlan mappings.

    Args:
        cache_dir: Directory to store profile YAML files.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def lookup(
        self,
        hw: HardwareProfile,
        model: ModelProfile,
        workload: WorkloadConfig,
    ) -> RunPlan | None:
        """Look up a cached plan for this (hw, model, workload) triple.

        Returns the cached RunPlan if found, or None.
        """
        key = self._hash_key(hw, model, workload)
        cache_file = self._cache_path(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict) or "plan" not in data:
                return None

            plan_data = data["plan"]
            plan = RunPlan(**{
                k: v for k, v in plan_data.items()
                if k in RunPlan.__dataclass_fields__
            })
            logger.info("Profile cache hit: %s", key[:12])
            return plan

        except Exception as e:
            logger.warning("Failed to read profile cache %s: %s", cache_file, e)
            return None

    def store(
        self,
        hw: HardwareProfile,
        model: ModelProfile,
        workload: WorkloadConfig,
        plan: RunPlan,
        metrics: InferenceMetrics | None = None,
    ) -> Path:
        """Store a tuned plan in the cache.

        Args:
            hw: Hardware profile.
            model: Model profile.
            workload: Workload configuration.
            plan: The selected RunPlan.
            metrics: Optional performance metrics to store alongside.

        Returns:
            Path to the cache file.
        """
        key = self._hash_key(hw, model, workload)
        cache_file = self._cache_path(key)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        from dataclasses import asdict

        entry = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hardware_fingerprint": hw.fingerprint(),
            "model_fingerprint": model.fingerprint(),
            "model_id": model.model_id,
            "model_path": model.model_path,
            "workload_type": workload.workload_type,
            "context_length": workload.context_length,
            "plan": asdict(plan),
        }

        if metrics:
            entry["metrics"] = {
                "tpot_ms": metrics.tpot_ms,
                "ttft_ms": metrics.ttft_ms,
                "output_tokens_per_s": metrics.output_tokens_per_s,
                "peak_vram_bytes": metrics.peak_vram_bytes,
                "peak_ram_bytes": metrics.peak_ram_bytes,
            }

        with open(cache_file, "w") as f:
            yaml.dump(entry, f, default_flow_style=False, sort_keys=False)

        logger.info("Stored profile cache: %s → %s", key[:12], cache_file)
        return cache_file

    def invalidate(
        self,
        hw: HardwareProfile,
        model: ModelProfile,
        workload: WorkloadConfig,
    ) -> bool:
        """Remove a cached entry. Returns True if it existed."""
        key = self._hash_key(hw, model, workload)
        cache_file = self._cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False

    def list_entries(self) -> list[dict]:
        """List all cached profile entries with summary info."""
        entries = []
        if not self.cache_dir.exists():
            return entries

        for f in sorted(self.cache_dir.glob("*.yaml")):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                entries.append({
                    "key": data.get("key", f.stem),
                    "model_id": data.get("model_id", "?"),
                    "workload": data.get("workload_type", "?"),
                    "created_at": data.get("created_at", "?"),
                    "file": str(f),
                })
            except Exception:
                continue

        return entries

    def clear(self) -> int:
        """Remove all cached profiles. Returns count of files removed."""
        count = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.yaml"):
                f.unlink()
                count += 1
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hash_key(
        self,
        hw: HardwareProfile,
        model: ModelProfile,
        workload: WorkloadConfig,
    ) -> str:
        """Generate a deterministic hash key from the triple."""
        components = [
            hw.fingerprint(),
            model.fingerprint(),
            f"{workload.workload_type}:{workload.context_length}:{workload.batch_size}",
        ]
        raw = "|".join(components)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the filesystem path for a cache key."""
        return self.cache_dir / f"{key[:16]}.yaml"

