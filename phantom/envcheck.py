"""Phase 0 environment check: Ollama, local models, TabPFN token, optional keys.

Never prints secret values, only whether they are set.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from phantom import config, runconfig


def ollama_status(host: str, timeout: float = 5.0) -> dict[str, Any]:
    """Return Ollama reachability, version and installed models from ``/api/tags``."""
    try:
        version = httpx.get(f"{host}/api/version", timeout=timeout).json().get("version")
        tags = httpx.get(f"{host}/api/tags", timeout=timeout).json().get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}", "models": []}
    models = [
        {
            "name": m.get("name"),
            "digest": m.get("digest"),
            "family": m.get("details", {}).get("family"),
            "parameter_size": m.get("details", {}).get("parameter_size"),
            "quantization": m.get("details", {}).get("quantization_level"),
            "size_bytes": m.get("size"),
        }
        for m in tags
    ]
    return {"reachable": True, "version": version, "models": models}


def gpu_name() -> str | None:
    """Return the NVIDIA GPU name and memory if ``nvidia-smi`` is available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return out or None


def collect() -> dict[str, Any]:
    """Gather the environment snapshot (no secret values)."""
    s = config.get_settings()
    return {
        "checked_at": runconfig.utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_name(),
        "ollama_host": s.ollama_host,
        "ollama": ollama_status(s.ollama_host),
        "tabpfn_token_set": s.tabpfn_token is not None,
        "tabpfn_no_browser": s.tabpfn_no_browser,
        "paperinstruments_key_set": s.paperinstruments_api_key is not None,
        "user_agent": s.user_agent,
    }


def main() -> int:
    """Print the environment report, save it to run_config.json, return exit code."""
    console = Console()
    env = collect()
    runconfig.update("env", env)

    ok = "[green]ok[/]"
    table = Table(title="Phantom Packages: environment check")
    table.add_column("Check")
    table.add_column("Status")
    table.add_row("Python", env["python"])
    table.add_row("GPU", env["gpu"] or "none detected")
    ollama = env["ollama"]
    table.add_row(
        f"Ollama ({env['ollama_host']})",
        f"{ok} v{ollama['version']}" if ollama["reachable"] else f"[red]unreachable[/] {ollama['error']}",
    )
    table.add_row("TABPFN_TOKEN", ok if env["tabpfn_token_set"] else "[red]missing[/] (needed for Phase 5)")
    table.add_row("TABPFN_NO_BROWSER=1", ok if env["tabpfn_no_browser"] else "[yellow]not set[/]")
    table.add_row(
        "PAPERINSTRUMENTS_API_KEY",
        ok if env["paperinstruments_key_set"] else "[yellow]not set[/] (Ultramarine arm will be skipped)",
    )
    table.add_row("User-Agent", env["user_agent"])
    console.print(table)

    if ollama["models"]:
        models = Table(title="Ollama models")
        for col in ("name", "family", "params", "quant", "digest"):
            models.add_column(col)
        for m in ollama["models"]:
            models.add_row(
                m["name"], m["family"] or "?", m["parameter_size"] or "?",
                m["quantization"] or "?", (m["digest"] or "")[:12],
            )
        console.print(models)
        families = {m["family"] for m in ollama["models"]}
        if len(families) < 3:
            console.print(
                f"[yellow]Only {len(families)} model families installed; the spec wants 3 "
                "different families.[/]"
            )
    console.print(f"Saved env snapshot to {config.RUN_CONFIG_PATH.relative_to(config.ROOT)}")
    return 0 if ollama["reachable"] else 1


if __name__ == "__main__":
    sys.exit(main())
