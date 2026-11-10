"""Model Resolver — resolve model identifiers to local GGUF file paths.

Handles three input forms:
  1. Local path:    /path/to/model.gguf
  2. HuggingFace:   Qwen/Qwen2.5-14B-Instruct-GGUF
  3. HF with file:  Qwen/Qwen2.5-14B-Instruct-GGUF:q4_k_m

For HuggingFace repos, downloads the GGUF file to a local cache and
returns the cached path. When multiple GGUF files exist in a repo,
the user can specify a quantization hint after a colon, or the resolver
picks the best match interactively.
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
        Absolute path to the local GGUF file.

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


def _download_from_hf(
    repo_id: str,
    quant_hint: str | None,
    cache_dir: Path,
    interactive: bool,
) -> str:
    """Download a GGUF model from HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
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

    # Select the right GGUF file
    target_file = _select_gguf_file(gguf_files, quant_hint, interactive)

    console.print(f"[bold green]Selected:[/bold green] {target_file}")

    # Check if already cached
    cached_path = cache_dir / repo_id.replace("/", "--") / target_file
    if cached_path.exists():
        console.print(f"[dim]Using cached file: {cached_path}[/dim]")
        return str(cached_path)

    # Download
    console.print(f"[bold]Downloading[/bold] {target_file} from {repo_id}...")
    console.print("[dim]This may take a while for large models.[/dim]")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=target_file,
            local_dir=str(cache_dir / repo_id.replace("/", "--")),
            local_dir_use_symlinks=False,
        )
        console.print(f"[bold green]Downloaded to:[/bold green] {downloaded_path}")
        return downloaded_path
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download '{target_file}' from '{repo_id}': {e}"
        )


def _select_gguf_file(
    gguf_files: list[str],
    quant_hint: str | None,
    interactive: bool,
) -> str:
    """Select one GGUF file from a list, using quant hint or user input.

    Selection priority:
      1. Exact quant_hint match in filename
      2. If no hint and only one file, use it
      3. If no hint and multiple files, pick best from preference list or ask user
    """
    if len(gguf_files) == 1:
        return gguf_files[0]

    # Normalize filenames for matching
    def extract_quant(filename: str) -> str | None:
        """Extract quantization type from a GGUF filename."""
        name_lower = filename.lower()
        # Try matching known quant patterns
        for qt in QUANT_PREFERENCE:
            if qt.replace("_", "-") in name_lower or qt in name_lower:
                return qt
        # Try regex for patterns like Q4_K_M, q4_k_m, Q8_0, etc.
        match = re.search(r'[qQ](\d+)[_-]?([kK])?[_-]?([sSmMlL])?', name_lower)
        if match:
            return match.group(0).lower().replace("-", "_")
        return None

    # If quant hint provided, find matching file
    if quant_hint:
        hint_normalized = quant_hint.lower().replace("-", "_")
        for f in gguf_files:
            file_quant = extract_quant(f)
            if file_quant and hint_normalized in file_quant:
                return f
            # Also try substring match on filename
            if hint_normalized in f.lower().replace("-", "_"):
                return f

        # No exact match — show available options
        console.print(f"[yellow]No file matching quant hint '{quant_hint}' found.[/yellow]")
        console.print("Available GGUF files:")
        for i, f in enumerate(gguf_files, 1):
            console.print(f"  {i}. {f}")
        raise ValueError(
            f"Quant hint '{quant_hint}' did not match any file in the repo. "
            f"Available: {', '.join(gguf_files)}"
        )

    # No hint — try auto-selecting from preference order
    file_quants = {f: extract_quant(f) for f in gguf_files}
    for preferred in QUANT_PREFERENCE:
        for f, qt in file_quants.items():
            if qt == preferred:
                console.print(
                    f"[dim]Auto-selected {preferred} quantization "
                    f"(use repo:quant_hint to override)[/dim]"
                )
                return f

    # Could not auto-select — ask user or pick first
    if interactive:
        return _interactive_select(gguf_files)

    # Non-interactive fallback: pick the smallest file
    console.print("[dim]Auto-selecting smallest GGUF file[/dim]")
    return gguf_files[0]


def _interactive_select(gguf_files: list[str]) -> str:
    """Prompt user to select a GGUF file interactively."""
    console.print()
    console.print("[bold]Multiple GGUF files found. Select one:[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=4)
    table.add_column("Filename")

    for i, f in enumerate(gguf_files, 1):
        table.add_row(str(i), f)

    console.print(table)

    while True:
        try:
            choice = console.input(
                f"\n[bold cyan]Select file (1-{len(gguf_files)}): [/bold cyan]"
            )
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(gguf_files):
                return gguf_files[idx]
            console.print("[red]Invalid selection. Try again.[/red]")
        except (ValueError, EOFError):
            console.print("[red]Please enter a number.[/red]")


def list_cached_models(cache_dir: Path | None = None) -> list[dict[str, str]]:
    """List all models currently in the local cache.

    Returns a list of dicts with 'repo', 'file', and 'path' keys.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    models = []

    if not cache.exists():
        return models

    for repo_dir in sorted(cache.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_name = repo_dir.name.replace("--", "/")
        for gguf_file in sorted(repo_dir.rglob("*.gguf")):
            models.append({
                "repo": repo_name,
                "file": gguf_file.name,
                "path": str(gguf_file),
                "size_gb": f"{gguf_file.stat().st_size / 1e9:.2f}",
            })

    return models

