"""Claude Code PreToolUse hook: stdin JSON -> exit code + stderr (BUILD_SPEC §9.3).

The guard is replaced by a fake, so no network, no model and no real ghost names.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field

import pytest

from phantom import config

HOOK_PATH = config.ROOT / "hooks" / "claude_pretooluse_guard.py"
_spec = importlib.util.spec_from_file_location("claude_pretooluse_guard", HOOK_PATH)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

GHOST = "zzq-ghostlib"  # synthetic


@dataclass
class FakeVerdict:
    name: str
    verdict: str
    p_ghost: float | None = None
    reasons: list[str] = field(default_factory=list)
    nearest: str | None = None
    nearest_lev: int | None = None
    checked_online: bool = True

    @property
    def normalized(self) -> str:
        return self.name.lower()


class FakeGuard:
    """Stands in for phantom.guard: fixed verdicts, records calls."""

    def __init__(self, verdicts=(), names=None, error: Exception | None = None):
        self.verdicts = list(verdicts)
        self.names = names
        self.error = error
        self.calls: list[dict] = []

    def names_from_command(self, cmd: str) -> list[str]:
        if self.names is not None:
            return self.names
        return [v.name for v in self.verdicts]

    def check(self, names, offline=False, prob="auto"):
        self.calls.append({"names": names, "offline": offline, "prob": prob})
        if self.error:
            raise self.error
        return self.verdicts


class ExplodingGuard:
    def __getattr__(self, item):
        raise AssertionError("guard must not be used for this input")


def payload(command: str, tool: str = "Bash") -> dict:
    return {"session_id": "s", "hook_event_name": "PreToolUse", "tool_name": tool,
            "tool_input": {"command": command}, "tool_use_id": "toolu_x"}


def run(p: dict | str, guard, **env) -> tuple[int, str]:
    text = p if isinstance(p, str) else json.dumps(p)
    return hook.run(text, env, guard=guard)


def test_non_shell_tool_is_allowed_without_touching_the_guard():
    p = {"tool_name": "Read", "tool_input": {"file_path": "x.py"}}
    assert run(p, ExplodingGuard()) == (0, "")


def test_command_without_installs_is_allowed_and_not_checked():
    g = FakeGuard(names=[])
    assert run(payload("ls -la && git status"), g) == (0, "")
    assert g.calls == []


def test_empty_or_missing_command_is_allowed():
    assert run({"tool_name": "Bash", "tool_input": {}}, ExplodingGuard()) == (0, "")
    assert run(payload("   "), ExplodingGuard()) == (0, "")


def test_ghost_blocks_with_the_spec_message():
    g = FakeGuard([FakeVerdict(GHOST, "GHOST", p_ghost=0.97, nearest="ghostlib", nearest_lev=4,
                               reasons=["404 on PyPI"])])
    code, msg = run(payload(f"pip install {GHOST}"), g)
    assert code == 2
    assert msg == (f"phantom-packages: blocked `pip install {GHOST}`: {GHOST} does not exist on PyPI "
                   f"(P(ghost)=0.97). Did you mean `ghostlib`? Never install unverified names.")
    assert g.calls == [{"names": [GHOST], "offline": False, "prob": "auto"}]


@pytest.mark.parametrize("nearest,lev,expected", [
    ("ghostlib", 4, "ghostlib"),        # affix: the name contains the real one
    ("zzq-ghostlab", 1, "zzq-ghostlab"),  # one typo away
    ("requests", 11, None),             # nearest exists but is unrelated: no "did you mean"
    (None, None, None),
])
def test_did_you_mean_only_when_plausible(nearest, lev, expected):
    v = FakeVerdict(GHOST, "GHOST", nearest=nearest, nearest_lev=lev)
    assert hook.plausible_suggestion(v) == expected
    assert ("Did you mean" in hook.ghost_message(v)) == (expected is not None)


def test_ghost_message_without_probability_or_nearest():
    g = FakeGuard([FakeVerdict(GHOST, "GHOST")])
    code, msg = run(payload(f"uv add {GHOST}"), g)
    assert code == 2
    assert msg == (f"phantom-packages: blocked `pip install {GHOST}`: {GHOST} does not exist on PyPI. "
                   "Never install unverified names.")


def test_powershell_tool_is_checked_too():
    g = FakeGuard([FakeVerdict(GHOST, "GHOST", p_ghost=0.9)])
    code, _ = run(payload(f"py -m pip install {GHOST}", tool="PowerShell"), g)
    assert code == 2


def test_suspect_blocks_with_reasons():
    g = FakeGuard([FakeVerdict("fresh-pkg", "SUSPECT", p_ghost=0.8,
                               reasons=["1 release", "first release 2026-10-01 (0 months ago)"])])
    code, msg = run(payload("pip install fresh-pkg"), g)
    assert code == 2
    assert "fresh-pkg" in msg and "1 release; first release 2026-10-01 (0 months ago)" in msg


def test_ok_is_allowed_silently():
    g = FakeGuard([FakeVerdict("requests", "OK", p_ghost=0.01)])
    assert run(payload("pip install requests"), g) == (0, "")


def test_one_ghost_among_real_names_blocks_and_names_only_the_ghost():
    g = FakeGuard([FakeVerdict("requests", "OK"), FakeVerdict(GHOST, "GHOST", p_ghost=0.9)])
    code, msg = run(payload(f"pip install requests {GHOST}"), g)
    assert code == 2 and msg.count("blocked") == 1 and f"`pip install {GHOST}`" in msg


def test_pypi_unreachable_fails_open_with_a_warning():
    v = FakeVerdict(GHOST, "GHOST_LIKELY", p_ghost=0.8, checked_online=False,
                    reasons=["PyPI unreachable: offline estimate"])
    code, msg = run(payload(f"pip install {GHOST}"), FakeGuard([v]))
    assert code == 0 and "warning" in msg and GHOST in msg


def test_pypi_unreachable_blocks_when_fail_closed():
    v = FakeVerdict(GHOST, "GHOST_LIKELY", p_ghost=0.8, checked_online=False)
    code, msg = run(payload(f"pip install {GHOST}"), FakeGuard([v]), PHANTOM_FAIL_CLOSED="1")
    assert code == 2 and GHOST in msg


def test_unverified_ok_blocks_only_when_fail_closed():
    v = FakeVerdict("requests", "OK_UNVERIFIED", p_ghost=0.02, checked_online=False)
    assert run(payload("pip install requests"), FakeGuard([v]))[0] == 0
    assert run(payload("pip install requests"), FakeGuard([v]), PHANTOM_FAIL_CLOSED="1")[0] == 2


def test_explicit_offline_mode_blocks_ghost_likely():
    v = FakeVerdict(GHOST, "GHOST_LIKELY", p_ghost=0.8, checked_online=False)
    g = FakeGuard([v])
    code, msg = run(payload(f"pip install {GHOST}"), g, PHANTOM_OFFLINE="1")
    assert code == 2 and "P(ghost)=0.80" in msg
    assert g.calls[0]["offline"] is True


def test_explicit_offline_mode_allows_ok_unverified():
    v = FakeVerdict("requests", "OK_UNVERIFIED", p_ghost=0.02, checked_online=False)
    assert run(payload("pip install requests"), FakeGuard([v]), PHANTOM_OFFLINE="1")[0] == 0


def test_guard_crash_fails_open_unless_fail_closed():
    g = FakeGuard(names=[GHOST], error=RuntimeError("boom"))
    code, msg = run(payload(f"pip install {GHOST}"), g)
    assert code == 0 and "warning" in msg
    code, msg = run(payload(f"pip install {GHOST}"), FakeGuard(names=[GHOST], error=RuntimeError("boom")),
                    PHANTOM_FAIL_CLOSED="1")
    assert code == 2 and "PHANTOM_FAIL_CLOSED" in msg


@pytest.mark.parametrize("text", ["not json", "", "[1, 2]", '{"tool_name": "Bash", "tool_input": "x"}'])
def test_malformed_stdin_fails_open_unless_fail_closed(text):
    assert run(text, ExplodingGuard())[0] == 0
    assert run(text, ExplodingGuard(), PHANTOM_FAIL_CLOSED="1")[0] == 2


def test_script_end_to_end_allows_non_shell_tool():
    """Run the hook as Claude Code would: a subprocess with JSON on stdin."""
    p = {"tool_name": "Read", "tool_input": {"file_path": "x"}}
    r = subprocess.run([sys.executable, str(HOOK_PATH)], input=json.dumps(p), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and r.stderr == ""


def test_script_end_to_end_no_install_command_skips_the_model():
    """A command without installs exits 0 and never imports torch/tabpfn (keeps the hook fast)."""
    pytest.importorskip("phantom.guard")
    code = (
        "import importlib.util, json, sys\n"
        f"s = importlib.util.spec_from_file_location('h', r'{HOOK_PATH}')\n"
        "h = importlib.util.module_from_spec(s); s.loader.exec_module(h)\n"
        "p = {'tool_name': 'Bash', 'tool_input': {'command': 'git status && ls -la'}}\n"
        "print(json.dumps([h.run(json.dumps(p), {}), 'torch' in sys.modules, 'tabpfn' in sys.modules]))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       cwd=str(config.ROOT))
    assert r.returncode == 0, r.stderr
    result, torch_loaded, tabpfn_loaded = json.loads(r.stdout.strip().splitlines()[-1])
    assert result == [0, ""] and not torch_loaded and not tabpfn_loaded
