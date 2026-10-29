"""``phantom`` command line (BUILD_SPEC §9.2): check | why | scan-cmd | stats.

Exit codes: 0 all OK, 1 SUSPECT present, 2 GHOST (or GHOST_LIKELY offline) present, 3 usage error.
Click's own usage errors normally exit 2; here they exit 3 so that 2 always means "ghost".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from typer._click import exceptions as click  # Typer >= 0.27 vendors Click; its exceptions live here
from typer.core import TyperGroup

from phantom import config, guard
from phantom.extract import _VALID

USAGE_ERROR = 3
console = Console(highlight=False)


class _Group(TyperGroup):
    """Typer group whose usage errors exit with ``USAGE_ERROR`` instead of Click's 2."""

    def main(self, *args: Any, standalone_mode: bool = True, **kwargs: Any) -> Any:
        if not standalone_mode:
            return super().main(*args, standalone_mode=False, **kwargs)
        try:
            rv = super().main(*args, standalone_mode=False, **kwargs)
        except click.UsageError as e:
            e.show()
            sys.exit(USAGE_ERROR)
        except click.ClickException as e:
            e.show()
            sys.exit(e.exit_code)
        except typer.Abort:
            click.echo("Aborted!", err=True)
            sys.exit(1)
        sys.exit(rv if isinstance(rv, int) else 0)


app = typer.Typer(cls=_Group, add_completion=False, no_args_is_help=True,
                  help="Catch pip packages that don't exist before you install them.")


def _usage(message: str) -> typer.Exit:
    console.print(f"[red]error:[/] {message}")
    return typer.Exit(USAGE_ERROR)


def _collect(args: list[str]) -> list[str]:
    """Names from each arg: a requirements/pyproject file if it exists, else a package name."""
    names: list[str] = []
    for arg in args:
        path = Path(arg)
        if path.is_file():
            try:
                names += guard.names_from_file(path)
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                raise _usage(f"cannot read {arg}: {exc}") from exc
        elif arg.endswith((".txt", ".toml")) or "/" in arg or "\\" in arg:
            raise _usage(f"file not found: {arg}")
        elif _VALID.match(arg):
            names.append(arg)
        else:
            raise _usage(f"not a package name or file: {arg!r}")
    return names


def _fmt_p(v: guard.Verdict) -> str:
    return "-" if v.p_ghost is None else f"{v.p_ghost:.2f}"


def _fmt_band(v: guard.Verdict) -> str:
    return "-" if v.band is None else f"{v.band[0]:.2f}-{v.band[1]:.2f}"


_STYLE = {"OK": "green", "OK_UNVERIFIED": "yellow", "SUSPECT": "yellow", "GHOST_LIKELY": "red", "GHOST": "bold red"}


def _report(verdicts: list[guard.Verdict], as_json: bool) -> None:
    code = guard.exit_code(verdicts)
    if as_json:
        typer.echo(json.dumps({"verdicts": [v.to_dict() for v in verdicts], "exit_code": code}, indent=2))
    else:
        table = Table()
        for col in ("Name", "Verdict", "P(ghost)", "Band", "Reasons"):
            table.add_column(col, justify="right" if col in ("P(ghost)", "Band") else "left")
        for v in verdicts:
            table.add_row(v.name, f"[{_STYLE[v.verdict]}]{v.verdict}[/]", _fmt_p(v), _fmt_band(v), " · ".join(v.reasons))
        console.print(table)
    raise typer.Exit(code)


@app.command()
def check(
    args: list[str] = typer.Argument(None, help="Package names, requirements*.txt or pyproject.toml files."),
    offline: bool = typer.Option(False, "--offline", help="Don't ask PyPI; use the TabPFN guard model only."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
    band: bool = typer.Option(False, "--band", help="Add the 5 fold models' min-max band (slower)."),
) -> None:
    """Check package names or dependency files. Exit 0 OK, 1 SUSPECT, 2 GHOST, 3 usage error."""
    if not args:
        raise _usage("give at least one package name or file")
    names = _collect(args)
    if not names:
        console.print("no dependencies found")
        raise typer.Exit(0)
    _report(guard.check(names, offline=offline, band=band), as_json)


@app.command("scan-cmd")
def scan_cmd(
    command: str = typer.Argument(..., help="A shell command, e.g. \"pip install requests\"."),
    offline: bool = typer.Option(False, "--offline", help="Don't ask PyPI; use the TabPFN guard model only."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Find pip/uv installs in a shell command and check them (used by the Claude Code hook)."""
    names = guard.names_from_command(command)
    if not names:
        if as_json:
            typer.echo(json.dumps({"verdicts": [], "exit_code": 0}))
        else:
            console.print("no installs found")
        raise typer.Exit(0)
    _report(guard.check(names, offline=offline), as_json)


@app.command()
def why(
    name: str = typer.Argument(..., help="One package name."),
    offline: bool = typer.Option(False, "--offline", help="Don't ask PyPI."),
) -> None:
    """A detailed card for one name: registry facts, nearest real name, model score and band."""
    if not _VALID.match(name):
        raise _usage(f"not a package name: {name!r}")
    [v] = guard.check([name], offline=offline, band=True, prob="always")
    card = guard.describe(v)
    console.rule(f"{v.name}  [{_STYLE[v.verdict]}]{v.verdict}[/]")
    for label, value in card.items():
        console.print(f"  [bold]{label:<22}[/] {value}")
    console.print("  [bold]reasons[/]")
    for r in v.reasons:
        console.print(f"    - {r}")
    raise typer.Exit(0)


@app.command()
def stats() -> None:
    """Headline numbers from results/summary.json."""
    if not config.SUMMARY_PATH.exists():
        raise _usage(f"{config.SUMMARY_PATH} not found; run the pipeline first")
    s = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    rates, study = s.get("rates", {}), s.get("study", {})
    lines: list[str] = []
    if o := rates.get("overall"):
        lines.append(f"ghost rate: {o['ghost_rate']:.1%} (95% CI {o['rate_lo']:.1%}-{o['rate_hi']:.1%}) "
                     f"of {o['mentions']:,} install suggestions; {o['gen_share']:.1%} of replies had >=1 ghost")
    for model, r in rates.get("per_model", {}).items():
        lines.append(f"  {model}: {r['ghost_rate']:.1%} ({r['ghosts']}/{r['mentions']})")
    for level, r in rates.get("per_niche_level", {}).items():
        lines.append(f"  niche {level}: {r['ghost_rate']:.1%}")
    if study:
        lines.append(f"study: {study.get('n_models')} models x {study.get('n_prompts')} prompts x "
                     f"{study.get('n_samples')} samples = {study.get('n_generations'):,} generations")
    if rep := s.get("repeat"):
        lines.append(f"ghost names: {rep['n_unique_ghost_names']}; repeated >=2x: {rep['repeated_2plus']} "
                     f"({rep['repeated_2plus_share']:.1%}); by >=2 models: {rep['multi_model']}")
    if a := s.get("eval", {}).get("A", {}).get("tabpfn"):
        lines.append(f"TabPFN AUROC (by prompt): {a['auroc']:.3f} ({a['auroc_lo']:.3f}-{a['auroc_hi']:.3f})")
    if t := s.get("threshold"):
        lines.append(f"threshold tau={t['tau']:.3f}: recall {t['recall']:.1%}, precision {t['precision']:.1%}")
    if g := s.get("guard_model"):
        o, gt = g["oof_split_a"], g["threshold"]
        lines.append(f"guard model ({g['model'].split(' (')[0]}, name features only): AUROC {o['auroc']:.3f} "
                     f"({o['auroc_lo']:.3f}-{o['auroc_hi']:.3f}); tau={gt['tau']:.3f}: recall {gt['recall']:.1%}, "
                     f"precision {gt['precision']:.1%}")
    if sq := s.get("squat"):
        lines.append(f"squat check: {sq['n_still_unregistered']} ghosts still unregistered, "
                     f"{sq['n_registered_during_study']} registered during the study, "
                     f"{sq['n_possible_slopsquat']} registered names match the squat rule")
    for line in lines:
        console.print(line)
    raise typer.Exit(0)
