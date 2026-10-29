"""Claude Code PreToolUse hook: block pip/uv installs of packages that don't exist (BUILD_SPEC §9.3).

Hook contract, verified 2026-10-30 against https://code.claude.com/docs/en/hooks:
- stdin is one JSON object: common fields session_id, transcript_path, cwd, permission_mode,
  hook_event_name ("PreToolUse"), plus tool_name, tool_input and tool_use_id. For the Bash tool the
  command is tool_input.command. The PowerShell tool (Windows): "The fields match the Bash tool, with
  the command string in `command`".
- Matcher: a string of only letters, digits, `_`, `-`, spaces, `,` and `|` is a list of exact names,
  so "Bash|PowerShell" matches both tools ("covers the PowerShell tool as well as Bash").
- Exit 2 blocks the tool call ("PreToolUse | Yes | Blocks the tool call"); "Claude sees the stderr
  message as the denial reason". Exit 0 with no output: no decision, the call continues through the
  normal permission flow, and "Claude never sees stderr from a hook that exits 0". Any other exit
  code is a non-blocking error.
- JSON alternative (not used here): exit 0 and print {"hookSpecificOutput": {"hookEventName":
  "PreToolUse", "permissionDecision": "allow|deny|ask|defer", "permissionDecisionReason": "..."}}.
- Command hooks time out after 600 s by default, and a timed-out PreToolUse hook does not block.

Policy: fail closed on GHOST/SUSPECT, fail open (with a warning) when PyPI can't be reached or the
guard itself fails, unless PHANTOM_FAIL_CLOSED=1. PHANTOM_OFFLINE=1 skips PyPI and blocks names the
offline TabPFN estimate calls ghost-likely. Commands without installs exit 0 without loading the model.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, when run as a plain script

SHELL_TOOLS = {"Bash", "PowerShell"}
PREFIX = "phantom-packages:"


def _p(v: Any) -> str:
    return f" (P(ghost)={v.p_ghost:.2f})" if v.p_ghost is not None else ""


def ghost_message(v: Any) -> str:
    """The §9.3 block message for a name PyPI does not know."""
    hint = f" Did you mean `{v.nearest}`?" if v.nearest and v.nearest != v.normalized else ""
    return (f"{PREFIX} blocked `pip install {v.name}`: {v.name} does not exist on PyPI{_p(v)}.{hint} "
            "Never install unverified names.")


def suspect_message(v: Any) -> str:
    """Block message for an existing name that matches the squat rules."""
    return (f"{PREFIX} blocked `pip install {v.name}`: {v.name} exists on PyPI but looks like a possible "
            f"slopsquat{_p(v)}: {'; '.join(v.reasons)}. Check it by hand before installing.")


def likely_message(v: Any) -> str:
    """Block message for an offline ghost estimate (PyPI not consulted)."""
    return (f"{PREFIX} blocked `pip install {v.name}`: PyPI was not checked and the offline estimate says "
            f"{v.name} is probably a ghost{_p(v)}. Verify it exists before installing.")


def decide(payload: Any, env: Mapping[str, str], guard: Any = None) -> tuple[int, str]:
    """Exit code and stderr text for one parsed hook payload."""
    if not isinstance(payload, dict):
        raise ValueError("hook input is not a JSON object")
    if payload.get("tool_name") not in SHELL_TOOLS:
        return 0, ""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        raise ValueError("tool_input is not an object")
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return 0, ""
    if guard is None:
        from phantom import guard  # light import; the model loads only if a verdict needs it

    names = guard.names_from_command(command)
    if not names:
        return 0, ""
    offline = env.get("PHANTOM_OFFLINE") == "1"
    fail_closed = env.get("PHANTOM_FAIL_CLOSED") == "1"
    blocks, warnings = [], []
    for v in guard.check(names, offline=offline, prob="auto"):
        if v.verdict == "GHOST":
            blocks.append(ghost_message(v))
        elif v.verdict == "SUSPECT":
            blocks.append(suspect_message(v))
        elif v.verdict == "GHOST_LIKELY":
            if offline or fail_closed:
                blocks.append(likely_message(v))
            else:
                warnings.append(f"{PREFIX} warning: PyPI unreachable; offline estimate says {v.name} is "
                                f"probably a ghost{_p(v)}. Allowed (fail-open); set PHANTOM_FAIL_CLOSED=1 to block.")
        elif v.verdict == "OK_UNVERIFIED" and not offline:
            if fail_closed:
                blocks.append(f"{PREFIX} blocked `pip install {v.name}`: PyPI could not be reached to verify "
                              f"it (PHANTOM_FAIL_CLOSED=1).")
            else:
                warnings.append(f"{PREFIX} warning: could not verify {v.name} on PyPI; allowed (fail-open).")
    if blocks:
        return 2, "\n".join(blocks)
    return 0, "\n".join(warnings)


def run(stdin_text: str, env: Mapping[str, str], guard: Any = None) -> tuple[int, str]:
    """Parse stdin and decide; malformed input or a guard failure fails open unless PHANTOM_FAIL_CLOSED=1."""
    try:
        return decide(json.loads(stdin_text), env, guard)
    except Exception as exc:  # noqa: BLE001 - a broken guard must never wedge the agent silently
        reason = f"{type(exc).__name__}: {exc}"
        if env.get("PHANTOM_FAIL_CLOSED") == "1":
            return 2, f"{PREFIX} guard error ({reason}); blocking because PHANTOM_FAIL_CLOSED=1."
        return 0, f"{PREFIX} warning: guard error ({reason}); allowed (fail-open)."


def main() -> None:
    """Entry point for Claude Code: JSON on stdin, verdict as exit code, message on stderr."""
    code, message = run(sys.stdin.buffer.read().decode("utf-8", errors="replace"), os.environ)
    if message:
        print(message, file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
