"""Model Resolver — resolve model identifiers to local GGUF file paths.

Handles three input forms:
  1. Local path:    /path/to/model.gguf
  2. HuggingFace:   Qwen/Qwen2.5-14B-Instruct-GGUF
  3. HF with file:  Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m

For HuggingFace repos, downloads the GGUF file to a local cache and
returns the cached path. Split GGUF models (sharded into multiple files)
are detected and all shards are downloaded automatically.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Default cache directory for downloaded models
DEFAULT_CACHE_DIR = Path.home() / ".adhie" / "models"

# Common quantization preference order (most preferred first for constrained hardware)
QUANT_PREFERENCE = [
    "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q3_k_m", "q3_k_l",
    "q6_k", "q8_0", "q2_k", "q3_k_s", "f16", "bf16", "f32",
    "iq4_xs", "iq4_nl", "iq3_s", "iq3_xxs", "iq2_xs", "iq2_xxs", "iq1_s",
]

# Regex to detect split GGUF shards: name-00001-of-00003.gguf
SPLIT_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def is_hf_identifier(model_ref: str) -> bool:
    """Check if a string looks like a HuggingFace model identifier.

    HF identifiers have the form 'org/model-name' or 'org/model-name:quant_hint'.
    Local paths contain slashes but typically start with /, ./, ~/, or have file extensions.
    """
    # Strip quant hint if present
    base = model_ref.split(":")[0]

    # Local path indicators
    if base.startswith(("/", "./", "../", "~")):
        return False
    if base.endswith(".gguf"):
        return False
    if os.path.exists(base):
        return False
    # Windows drive letter paths like D:\path or D:/path
    if len(base) >= 2 and base[1] == ":" and base[0].isalpha():
        return False

    # HF identifier: org/model-name (exactly one slash, no path separators beyond that)
    parts = base.split("/")
    if len(parts) == 2 and all(p and not p.startswith(".") for p in parts):
        return True

    return False


def parse_model_ref(model_ref: str) -> tuple[str, str | None]:
    """Parse a model reference into (repo_id_or_path, quant_hint).

    Examples:
        "Qwen/Qwen2.5-14B-Instruct-GGUF"         → ("Qwen/Qwen2.5-14B-Instruct-GGUF", None)
        "Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m"  → ("Qwen/Qwen2.5-14B-Instruct-GGUF", "q4_k_m")
        "/path/to/model.gguf"                      → ("/path/to/model.gguf", None)
    """
    if ":" in model_ref and is_hf_identifier(model_ref.split(":")[0]):
        repo_id, quant_hint = model_ref.rsplit(":", 1)
        return repo_id, quant_hint.lower().strip()
    return model_ref, None


def resolve_model(
    model_ref: str,
    cache_dir: Path | None = None,
    interactive: bool = True,
) -> str:
    """Resolve a model reference to a local GGUF file path.

    Args:
        model_ref: Local path, HuggingFace repo ID, or repo:quant_hint.
        cache_dir: Where to cache downloaded files. Defaults to ~/.adhie/models.
        interactive: If True, prompt user to select among multiple GGUF files.

    Returns:
        Absolute path to the local GGUF file. For split models, returns
        the path to shard 1 (llama-server auto-discovers sibling shards).

    Raises:
        FileNotFoundError: If model cannot be found or downloaded.
        ValueError: If repo has no GGUF files or selection fails.
    """
    base_ref, quant_hint = parse_model_ref(model_ref)

    if not is_hf_identifier(base_ref):
        # Local path — resolve and validate
        path = Path(base_ref).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if not path.suffix.lower() == ".gguf":
            raise ValueError(f"Expected a .gguf file, got: {path}")
        return str(path)

    # HuggingFace download
    return _download_from_hf(base_ref, quant_hint, cache_dir or DEFAULT_CACHE_DIR, interactive)


# ------------------------------------------------------------------
# Split GGUF helpers
# ------------------------------------------------------------------

def _parse_split_shard(filename: str) -> tuple[str, int, int] | None:
    """Parse a split shard filename, returning (stem, index, total) or None.

    Example: "model-q4_k_m-00001-of-00003.gguf" → ("model-q4_k_m", 1, 3)
    """
    m = SPLIT_SHARD_RE.match(filename)
    if m:
        return m.group("stem"), int(m.group("idx")), int(m.group("total"))
    return None


def _group_split_files(gguf_files: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Separate GGUF files into singles and split-file groups.

    Returns:
        (singles, split_groups) where:
          - singles: list of non-split GGUF filenames
          - split_groups: dict mapping stem → sorted list of shard filenames
    """
    singles: list[str] = []
    groups: dict[str, list[str]] = {}

    for f in gguf_files:
        # Use just the basename for pattern matching
        basename = f.split("/")[-1] if "/" in f else f
        parsed = _parse_split_shard(basename)
        if parsed:
            stem, _idx, _total = parsed
            groups.setdefault(stem, []).append(f)
        else:
            singles.append(f)

    # Sort shards within each group by index
    for stem in groups:
        groups[stem].sort()

    return singles, groups


def _shard1_of_group(shard_files: list[str]) -> str:
    """Return shard-1 filename from a sorted list of shards."""
    for f in shard_files:
        basename = f.split("/")[-1] if "/" in f else f
        parsed = _parse_split_shard(basename)
        if parsed and parsed[1] == 1:
            return f
    # Fallback: first file in sorted order
    return shard_files[0]


# ------------------------------------------------------------------
# HuggingFace download
# ------------------------------------------------------------------

def _download_from_hf(
    repo_id: str,
    quant_hint: str | None,
    cache_dir: Path,
    interactive: bool,
) -> str:
    """Download a GGUF model from HuggingFace Hub.

    For split GGUF models, downloads ALL shards and returns the
    path to shard 1 (llama-server auto-discovers the rest).
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )

    console.print(f"[bold blue]Resolving model:[/bold blue] {repo_id}")

    api = HfApi()

    # List all files in the repo to find GGUFs
    try:
        files = list(api.list_repo_tree(repo_id, repo_type="model"))
        gguf_files = [
            f.path for f in files
            if hasattr(f, "path") and f.path.endswith(".gguf")
        ]
    except Exception as e:
        raise FileNotFoundError(
            f"Could not access HuggingFace repo '{repo_id}': {e}"
        )

    if not gguf_files:
        raise ValueError(
            f"No GGUF files found in repo '{repo_id}'. "
            "Make sure the repo contains GGUF-formatted model files."
        )

    # Group into singles and split sets
    singles, split_groups = _group_split_files(gguf_files)

    # Select the right GGUF file (or split group)
    target_file, is_split, split_shards = _select_gguf_file(
        singles, split_groups, quant_hint, interactive,
    )

    console.print(f"[bold green]Selected:[/bold green] {target_file}")

    # Determine which files to download
    local_dir = cache_dir / repo_id.replace("/", "--")
    files_to_download = split_shards if is_split else [target_file]

    # Check if already cached (all shards must exist for split models)
    all_cached = all((local_dir / f).exists() for f in files_to_download)
    shard1_path = local_dir / target_file

    if all_cached:
        console.print(f"[dim]Using cached file: {shard1_path}[/dim]")
        return str(shard1_path)

    # Download all required files
    if is_split:
        console.print(
            f"[bold]Downloading[/bold] {len(files_to_download)} shard(s) "
            f"from {repo_id}..."
        )
    else:
        console.print(f"[bold]Downloading[/bold] {target_file} from {repo_id}...")
    console.print("[dim]This may take a while for large models.[/dim]")

    for fname in files_to_download:
        cached_path = local_dir / fname
        if cached_path.exists():
            console.print(f"  [dim]Shard cached: {fname}[/dim]")
            continue
        try:
            console.print(f"  [bold]↓[/bold] {fname}")
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download '{fname}' from '{repo_id}': {e}"
            )

    console.print(f"[bold green]Downloaded to:[/bold green] {shard1_path}")
    return str(shard1_path)


def _select_gguf_file(
    singles: list[str],
    split_groups: dict[str, list[str]],
    quant_hint: str | None,
    interactive: bool,
) -> tuple[str, bool, list[str]]:
    """Select one GGUF file (or split group) from the repo.

    Returns:
        (target_file, is_split, files_to_download) where:
          - target_file: the primary filename (shard-1 for splits)
          - is_split: whether this is a split model
          - files_to_download: list of all files to download
    """
    # Build a unified candidate list: each entry is (display_name, is_split, files, quant)
    candidates: list[tuple[str, bool, list[str], str | None]] = []

    for f in singles:
        qt = _extract_quant(f)
        candidates.append((f, False, [f], qt))

    for stem, shards in sorted(split_groups.items()):
        shard1 = _shard1_of_group(shards)
        qt = _extract_quant(stem)
        n = len(shards)
        candidates.append((shard1, True, shards, qt))

    if not candidates:
        raise ValueError("No GGUF files found.")

    # Single candidate — use it
    if len(candidates) == 1:
        c = candidates[0]
        return c[0], c[1], c[2]

    # If quant hint provided, find matching candidate
    if quant_hint:
        hint_normalized = quant_hint.lower().replace("-", "_")

        # First: try exact quant match
        for display, is_split, files, qt in candidates:
            if qt and hint_normalized == qt:
                return display, is_split, files

        # Second: try substring match on quant
        for display, is_split, files, qt in candidates:
            if qt and hint_normalized in qt:
                return display, is_split, files

        # Third: try substring match on filename
        for display, is_split, files, qt in candidates:
            if hint_normalized in display.lower().replace("-", "_"):
                return display, is_split, files

        # No match
        all_names = [c[0] for c in candidates]
        console.print(f"[yellow]No file matching quant hint '{quant_hint}' found.[/yellow]")
        console.print("Available GGUF files:")
        for i, name in enumerate(all_names, 1):
            console.print(f"  {i}. {name}")
        raise ValueError(
            f"Quant hint '{quant_hint}' did not match any file in the repo. "
            f"Available: {', '.join(all_names)}"
        )

    # No hint — try auto-selecting from preference order
    # Prefer single files over splits when same quant is available in both
    for preferred in QUANT_PREFERENCE:
        # Check singles first
        for display, is_split, files, qt in candidates:
            if qt == preferred and not is_split:
                console.print(
                    f"[dim]Auto-selected {preferred} quantization "
                    f"(use repo:quant_hint to override)[/dim]"
                )
                return display, is_split, files
        # Then splits
        for display, is_split, files, qt in candidates:
            if qt == preferred and is_split:
                console.print(
                    f"[dim]Auto-selected {preferred} quantization (split, "
                    f"{len(files)} shards; use repo:quant_hint to override)[/dim]"
                )
                return display, is_split, files

    # Could not auto-select — ask user or pick first
    if interactive:
        return _interactive_select(candidates)

    # Non-interactive fallback: pick first single, or first split
    c = candidates[0]
    return c[0], c[1], c[2]


def _extract_quant(filename: str) -> str | None:
    """Extract quantization type from a GGUF filename or stem."""
    name_lower = filename.lower()
    # Try matching known quant patterns (exact)
    for qt in QUANT_PREFERENCE:
        # Match with separators: -q4_k_m. or _q4_k_m_ etc.
        pattern_underscore = qt
        pattern_dash = qt.replace("_", "-")
        if pattern_underscore in name_lower or pattern_dash in name_lower:
            return qt
    # Try regex for patterns like Q4_K_M, q4_k_m, Q8_0, etc.
    match = re.search(r'[qQ](\d+)[_-]?([kK])?[_-]?([sSmMlL])?', name_lower)
    if match:
        return match.group(0).lower().replace("-", "_")
    return None


def _interactive_select(
    candidates: list[tuple[str, bool, list[str], str | None]],
) -> tuple[str, bool, list[str]]:
    """Prompt user to select a GGUF file interactively."""
    console.print()
    console.print("[bold]Multiple GGUF files found. Select one:[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=4)
    table.add_column("Filename")
    table.add_column("Quant")
    table.add_column("Type")

    for i, (display, is_split, files, qt) in enumerate(candidates, 1):
        ftype = f"split ({len(files)} shards)" if is_split else "single"
        table.add_row(str(i), display, qt or "?", ftype)

    console.print(table)

    while True:
        try:
            choice = console.input(
                f"\n[bold cyan]Select file (1-{len(candidates)}): [/bold cyan]"
            )
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                return c[0], c[1], c[2]
            console.print("[red]Invalid selection. Try again.[/red]")
        except (ValueError, EOFError):
            console.print("[red]Please enter a number.[/red]")


def list_cached_models(cache_dir: Path | None = None) -> list[dict[str, str]]:
    """List all models currently in the local cache.

    Returns a list of dicts with 'repo', 'file', and 'path' keys.
    Groups split shards under a single entry.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    models = []

    if not cache.exists():
        return models

    for repo_dir in sorted(cache.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_name = repo_dir.name.replace("--", "/")

        # Collect all GGUF files
        all_gguf = sorted(repo_dir.rglob("*.gguf"))

        # Group split shards
        seen_stems: set[str] = set()
        for gguf_file in all_gguf:
            parsed = _parse_split_shard(gguf_file.name)
            if parsed:
                stem, idx, total = parsed
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                # Count how many shards we have
                shard_count = sum(
                    1 for f in all_gguf
                    if _parse_split_shard(f.name) and _parse_split_shard(f.name)[0] == stem
                )
                total_size = sum(
                    f.stat().st_size for f in all_gguf
                    if _parse_split_shard(f.name) and _parse_split_shard(f.name)[0] == stem
                )
                models.append({
                    "repo": repo_name,
                    "file": f"{stem} ({shard_count}/{total} shards)",
                    "path": str(gguf_file),
                    "size_gb": f"{total_size / 1e9:.2f}",
                    "is_split": "true",
                    "shard_count": str(shard_count),
                    "total_shards": str(total),
                })
            else:
                models.append({
                    "repo": repo_name,
                    "file": gguf_file.name,
                    "path": str(gguf_file),
                    "size_gb": f"{gguf_file.stat().st_size / 1e9:.2f}",
                    "is_split": "false",
                })

    return models
