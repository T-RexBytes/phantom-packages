"""Measure the Claude Code hook's end-to-end latency (cold process, as Claude Code runs it).

Runs hooks/claude_pretooluse_guard.py through `uv run` with the same stdin JSON Claude Code
sends, for: a command with no install, a real install, and a ghost install (default mode and
PHANTOM_HOOK_PROB=1). Live PyPI metadata only; the ghost is an invented name, not a study name.
Writes results/hook_latency.json and summary.json["hook_latency"]. Not part of `tasks.py all`
(it depends on the network and the machine).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time

from phantom import config, runconfig

OUT = config.RESULTS / "hook_latency.json"
HOOK = config.ROOT / "hooks" / "claude_pretooluse_guard.py"
GHOST = "zzq-ghostlib-not-real"  # invented for this benchmark
CASES = {
    "no_install": ("git status && ls", {}),
    "real_install": ("pip install requests", {}),
    "ghost_install_default": (f"pip install {GHOST}", {}),
    "ghost_install_with_p_ghost": (f"pip install {GHOST}", {"PHANTOM_HOOK_PROB": "1"}),
}


def run_once(command: str, extra_env: dict[str, str]) -> tuple[float, int]:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    env = {**os.environ, **extra_env, "PYTHONIOENCODING": "utf-8"}
    env.pop("VIRTUAL_ENV", None)
    start = time.perf_counter()
    proc = subprocess.run(["uv", "run", "--project", str(config.ROOT), "python", str(HOOK)], input=payload,
                          capture_output=True, text=True, env=env, cwd=config.ROOT.parent)
    return time.perf_counter() - start, proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3)
    args, _ = ap.parse_known_args()
    run_once("git status", {})  # warm the uv environment once; not counted
    cases = {}
    for name, (command, env) in CASES.items():
        times, codes = zip(*(run_once(command, env) for _ in range(args.runs)))
        cases[name] = {"command": command, "env": env, "seconds": [round(t, 2) for t in times],
                       "median_s": round(statistics.median(times), 2), "exit_codes": sorted(set(codes))}
        print(f"{name:<28} median {cases[name]['median_s']:>6.2f} s  exit {cases[name]['exit_codes']}")
    result = {"measured_at": runconfig.utc_now(), "runs_per_case": args.runs,
              "machine": f"{platform.system()} {platform.release()}, CPU torch (no CUDA)", "cases": cases}
    runconfig.save(result, OUT)
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["hook_latency"] = result
    runconfig.save(summary, config.SUMMARY_PATH)


if __name__ == "__main__":
    main()
