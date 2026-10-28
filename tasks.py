"""Run a pipeline stage: ``uv run tasks.py <stage> [--fast]``.

``all`` runs every implemented stage in order. ``--fast`` is passed to each
stage script, which then reuses cached data instead of calling models.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

# stage name -> (script, phase that implements it)
STAGES: dict[str, tuple[str, int]] = {
    "prompts": ("01_prompts.py", 1),
    "summon": ("02_summon.py", 2),
    "extract": ("03_extract_label.py", 3),
    "features": ("04_features.py", 4),
    "eval": ("05_eval.py", 5),
    "squat": ("06_squat_check.py", 6),
    "figures": ("07_figures.py", 8),
    "site": ("08_build_site_data.py", 8),
}
UTILITY = {
    "env": "check Ollama, models, TabPFN token, optional keys",
    "test": "run pytest -q",
}


def run_script(name: str, extra: list[str]) -> int:
    """Run one experiment script with the current interpreter; return its exit code."""
    script, phase = STAGES[name]
    path = EXPERIMENTS / script
    if not path.exists():
        print(f"[{name}] not implemented yet (Phase {phase}): experiments/{script}")
        return 1
    print(f"[{name}] experiments/{script} {' '.join(extra)}".rstrip(), flush=True)
    return subprocess.call([sys.executable, str(path), *extra], cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a stage."""
    choices = [*UTILITY, *STAGES, "all"]
    parser = argparse.ArgumentParser(
        description="Phantom Packages pipeline",
        epilog="stages: " + ", ".join(choices),
    )
    parser.add_argument("stage", choices=choices, metavar="stage")
    parser.add_argument("--fast", action="store_true", help="reuse cached data; no model calls")
    args, extra = parser.parse_known_args(argv)
    if args.fast:
        extra = ["--fast", *extra]

    if args.stage == "env":
        from phantom import envcheck

        return envcheck.main()
    if args.stage == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", *extra], cwd=ROOT)
    if args.stage == "all":
        for name in STAGES:
            code = run_script(name, extra)
            if code != 0:
                print(f"[all] stopped at '{name}' (exit {code})")
                return code
        return 0
    return run_script(args.stage, extra)


if __name__ == "__main__":
    sys.exit(main())
