"""Ultramarine hosted arms (BUILD_SPEC §5.2): the same prompts through a hosted model.

Parts (all ``ultramarine-1-flash``, sequential, one request at a time):
  A. generation arm, every prompt x samples 1-2, ``reasoning_effort="low"`` (model label
     ``ultramarine-1-flash``) — the hosted comparison to the local open models;
  B. reasoning ablation, every prompt x sample 1, ``reasoning_effort="high"`` (model label
     ``ultramarine-1-flash@high``, never mixed with A);
  C. self-check baseline: the gemma prompt "does this package exist?" for every study name.
Hosted generations go to ``data/raw_generations_hosted.jsonl`` (never the local log).

Contract re-verified 2026-10-30 against the live docs:
- https://docs.paperinstruments.com/docs/ultramarine/chat-completions: OpenAI-compatible chat
  completions at ``https://api.paperinstruments.com/v1``; ``reasoning_effort`` is
  ``low|high|max`` (default ``max``; anything else -> 400); ``max_completion_tokens`` caps
  reasoning + answer together; ``temperature`` 0-2; ``seed``; ``logprobs`` and
  ``top_logprobs`` (integer 0-20) — the logprobs response shape is not documented, so it is
  probed and parsed defensively; ``message.reasoning_content`` carries the reasoning; usage
  has ``prompt_tokens_details.cached_tokens`` and ``completion_tokens_details.reasoning_tokens``.
- https://docs.paperinstruments.com/docs/ultramarine/rate-limits: ``x-ratelimit-*`` headers;
  "On 429, wait for retry-after"; "For higher limits, contact Paper Instruments".
- ``GET /v1/pricing`` (public) returns integer nanodollar rates; billing is prepaid and the
  worst-case cost of a call is reserved up front (docs/api_notes/tabpfn_ultramarine.md §2.1).
Spend is tracked in a persistent ledger with a hard cap, so restarts can never overspend.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from phantom import config, runconfig
from phantom.generate import generation_key, strip_thinking
from phantom.runconfig import utc_now

BASE_URL = "https://api.paperinstruments.com/v1"
MODEL = "ultramarine-1-flash"
MODEL_HIGH = f"{MODEL}@high"          # label for the reasoning-effort ablation (part B)
REASONING_EFFORT = "low"
# Reasoning tokens count against this cap; 350 visible tokens + reasoning headroom.
MAX_COMPLETION_TOKENS = 1024
MAX_COMPLETION_TOKENS_HIGH = 4096
SELFCHECK_MAX_COMPLETION_TOKENS = 256
SELFCHECK_TOP_LOGPROBS = 20
PROMPT_TOKENS_ESTIMATE = 150          # worst-case input estimate per generation call (~90 measured locally)
CAP_USD = 4.00                        # hard cap for all hosted spend (user budget $5, $1 headroom)
PRICING_PATH = config.RESULTS / "ultramarine_pricing.json"
HOSTED_RAW_PATH = config.DATA / "raw_generations_hosted.jsonl"
LEDGER_PATH = config.DATA / "hosted_spend.json"
SELFCHECK_CACHE = config.CACHE / "selfcheck_hosted"
MAX_TRIES = 6
MAX_CONSECUTIVE_FAILURES = 5


class BudgetExceeded(RuntimeError):
    """A call's worst case would push spend over the cap."""


def fetch_pricing() -> dict[str, Any]:
    """GET the public price card and save it to results/ultramarine_pricing.json."""
    card = httpx.get(f"{BASE_URL}/pricing", timeout=20).raise_for_status().json()
    card["fetched_at"] = utc_now()
    runconfig.save(card, PRICING_PATH)
    return card


def cost_usd(prompt_tokens: int, cached_tokens: int, completion_tokens: int, card: dict) -> float:
    """Cost of one call from the price card's integer nanodollar-per-token rates."""
    nano = (
        (prompt_tokens - cached_tokens) * card["uncached_input_nanodollars_per_token"]
        + cached_tokens * card["cached_input_nanodollars_per_token"]
        + completion_tokens * card["output_nanodollars_per_token"]
    )
    return nano / 1e9


def max_cost_usd(n_calls: int, prompt_tokens_each: int, card: dict,
                 max_completion: int = MAX_COMPLETION_TOKENS) -> float:
    """Upper bound on spend if every call used the full completion budget."""
    return n_calls * cost_usd(prompt_tokens_each, 0, max_completion, card)


class Ledger:
    """Persistent cumulative spend (``data/hosted_spend.json``) with a hard cap."""

    def __init__(self, path: Path = LEDGER_PATH, cap_usd: float = CAP_USD) -> None:
        self.path, self.cap_usd = path, cap_usd
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "spent_usd": 0.0, "calls": 0, "parts": {}}

    @property
    def spent(self) -> float:
        return float(self.data["spent_usd"])

    def check(self, worst_usd: float) -> None:
        """Raise BudgetExceeded unless spent + worst case stays within the cap."""
        if self.spent + worst_usd > self.cap_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f} + worst case ${worst_usd:.4f} > cap ${self.cap_usd:.2f}")

    def add(self, part: str, usd: float, usage: dict[str, int]) -> None:
        """Record one call and persist immediately (write-then-replace)."""
        p = self.data["parts"].setdefault(part, {"calls": 0, "usd": 0.0, "prompt_tokens": 0,
                                                 "completion_tokens": 0, "reasoning_tokens": 0})
        p["calls"] += 1
        p["usd"] += usd
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            p[k] += int(usage.get(k, 0) or 0)
        self.data["spent_usd"] = self.spent + usd
        self.data["calls"] += 1
        self.data["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def _usage(resp: Any) -> dict[str, int]:
    u = resp.usage
    cached = (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0
    reasoning = (u.completion_tokens_details.reasoning_tokens if u.completion_tokens_details else 0) or 0
    return {"prompt_tokens": u.prompt_tokens, "cached_tokens": cached,
            "completion_tokens": u.completion_tokens, "reasoning_tokens": reasoning}


def with_retry(call: Callable[[], Any], sleep: Callable[[float], None] = time.sleep,
               max_tries: int = MAX_TRIES) -> Any:
    """Retry 429/5xx/transport errors with exponential backoff, honouring Retry-After."""
    import openai

    for attempt in range(max_tries):
        try:
            return call()
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError,
                openai.APITimeoutError) as exc:
            if attempt == max_tries - 1:
                raise
            resp = getattr(exc, "response", None)
            headers = resp.headers if resp is not None else {}
            if str(headers.get("x-should-retry", "")).lower() == "false":
                raise
            try:
                wait = float(headers.get("retry-after", ""))
            except ValueError:
                wait = min(60.0, 2.0 ** (attempt + 1))
            sleep(wait)
    raise RuntimeError("unreachable")


def ultramarine_record(client: Any, prompt_id: str, text: str, card: dict, sample: int = 1,
                       effort: str = REASONING_EFFORT, max_completion: int = MAX_COMPLETION_TOKENS,
                       label: str = MODEL, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """One hosted generation as a ``raw_generations_hosted.jsonl`` row (``arm="hosted"``)."""
    seed = config.SAMPLE_SEED_BASE + sample
    start = time.perf_counter()
    resp = with_retry(lambda: client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=config.TEMPERATURE,
        max_completion_tokens=max_completion,
        reasoning_effort=effort,
        seed=seed,
    ), sleep=sleep)
    latency = time.perf_counter() - start
    usage = _usage(resp)
    choice = resp.choices[0]
    return {
        "key": generation_key(label, prompt_id, sample),
        "model": label,
        "digest": None,
        "prompt_id": prompt_id,
        "sample": sample,
        "seed": seed,
        "temperature": config.TEMPERATURE,
        "output": strip_thinking(choice.message.content or ""),
        "done_reason": choice.finish_reason,
        "latency_s": round(latency, 3),
        "eval_count": usage["completion_tokens"],
        "prompt_eval_count": usage["prompt_tokens"],
        "usage": usage,
        "cost_usd": cost_usd(usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"], card),
        "reasoning_effort": effort,
        "max_completion_tokens": max_completion,
        "ts": utc_now(),
        "arm": "hosted",
    }


def _positions(choice: Any) -> list | None:
    """OpenAI-style ``choice.logprobs.content`` -> ``[[token, logprob, [[alt, logprob], ...]], ...]``."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return None
    pos = [[t.token, t.logprob, [[a.token, a.logprob] for a in (t.top_logprobs or [])]] for t in content]
    # Probed 2026-10-30: the stream starts with the reasoning-close token "</think>"; the answer
    # distribution is at the first position after it.
    ends = [i for i, (tok, _lp, _alts) in enumerate(pos) if (tok or "").strip() == "</think>"]
    return pos[ends[-1] + 1:] if ends else pos


def selfcheck_record(client: Any, name: str, card: dict,
                     sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Ask the hosted model the gemma self-check question for one name (part C)."""
    from phantom import selfcheck

    resp = with_retry(lambda: client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": selfcheck.PROMPT.format(name=name)}],
        temperature=0.0,
        max_completion_tokens=SELFCHECK_MAX_COMPLETION_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        seed=selfcheck.SEED,
        logprobs=True,
        top_logprobs=SELFCHECK_TOP_LOGPROBS,
    ), sleep=sleep)
    usage = _usage(resp)
    choice = resp.choices[0]
    reply = choice.message.content or ""
    positions = _positions(choice)
    return {"name": name, "model": MODEL, "reply": reply, "answer": selfcheck.parse_answer(reply),
            "p_no": selfcheck.p_no(positions), "top_logprobs": (positions or [])[:3],
            "finish_reason": choice.finish_reason, "usage": usage,
            "cost_usd": cost_usd(usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"], card),
            "ts": utc_now()}


def selfcheck_cache_path(name: str) -> Path:
    from phantom.http import cache_key

    return SELFCHECK_CACHE / f"{cache_key(name)}.json"


def cached_selfcheck(names) -> dict[str, dict[str, Any]]:
    """Cached hosted self-check answers for ``names`` (no network)."""
    out = {}
    for n in names:
        p = selfcheck_cache_path(n)
        if p.exists():
            out[n] = json.loads(p.read_text(encoding="utf-8"))
    return out


def ask_selfcheck(names, client: Any, card: dict, ledger: Ledger,
                  progress: Callable[[int, int, float], None] | None = None) -> dict[str, dict[str, Any]]:
    """Answers for every name, calling the API only for uncached names, within the ledger cap."""
    names = sorted(set(names))
    out = cached_selfcheck(names)
    missing = [n for n in names if n not in out]
    worst = cost_usd(80, 0, SELFCHECK_MAX_COMPLETION_TOKENS, card)
    SELFCHECK_CACHE.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(missing, 1):
        ledger.check(worst)
        rec = selfcheck_record(client, n, card)
        ledger.add("C_selfcheck", rec["cost_usd"], rec["usage"])
        selfcheck_cache_path(n).write_text(json.dumps(rec), encoding="utf-8")
        out[n] = rec
        if progress:
            progress(i, len(missing), ledger.spent)
    return out


def client_from_env():
    """OpenAI client for Ultramarine, or None when no key is configured."""
    key = config.get_settings().paperinstruments_api_key
    if not key:
        return None
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=300.0)


# ---------------------------------------------------------------- analysis (stage 03 / 05)

def hosted_in_plan(records, prompt_ids) -> list[dict[str, Any]]:
    """Hosted rows for the planned prompts (local rows never come from the hosted file)."""
    ids = set(prompt_ids)
    return [r for r in records if r.get("arm") == "hosted" and r["prompt_id"] in ids]


def hosted_stats(lab, gens, records) -> dict[str, dict[str, Any]]:
    """Per hosted model label: ghost rate (subtypes excluded), generation share, tokens and cost."""
    import pandas as pd

    main = lab[lab["category"].isin(["real", "ghost"])]
    ghost_keys = set(main.loc[main["y_ghost"], "key"])
    rec = pd.DataFrame(records)
    out = {}
    for label, g in gens.groupby("model"):
        m = main[main["model"] == label]
        r = rec[rec["model"] == label]
        usage = pd.DataFrame(list(r["usage"]))
        out[label] = {
            "reasoning_effort": str(r["reasoning_effort"].iloc[0]),
            "generations": int(len(g)),
            "mentions": int(len(m)),
            "ghosts": int(m["y_ghost"].sum()),
            "ghost_rate": float(m["y_ghost"].mean()) if len(m) else float("nan"),
            "with_ghost": int(g["key"].isin(ghost_keys).sum()),
            "gen_share": float(g["key"].isin(ghost_keys).mean()),
            "unique_ghost_names": int(m.loc[m["y_ghost"], "name"].nunique()),
            "prompt_tokens": int(usage["prompt_tokens"].sum()),
            "completion_tokens": int(usage["completion_tokens"].sum()),
            "reasoning_tokens": int(usage["reasoning_tokens"].sum()),
            "cost_usd": float(r["cost_usd"].sum()),
            "cost_per_1k_generations_usd": float(r["cost_usd"].sum() / len(r) * 1000),
            "length_capped_share": float((r["done_reason"] == "length").mean()),
        }
    return out


def ablation(lab, gens, low: str = MODEL, high: str = MODEL_HIGH, sample: int = 1) -> dict[str, Any]:
    """Paired low vs high reasoning effort on the same prompts (sample 1 of each).

    Counts prompts where only one effort produced a ghost (McNemar-style discordant pairs).
    """
    main = lab[lab["category"].isin(["real", "ghost"])]
    g = gens[gens["sample"] == sample]
    ghost_keys = set(main.loc[main["y_ghost"], "key"])
    per = {}
    for label in (low, high):
        gg = g[g["model"] == label]
        per[label] = dict(zip(gg["prompt_id"], gg["key"].isin(ghost_keys)))
    common = sorted(set(per[low]) & set(per[high]))
    both = sum(per[low][p] and per[high][p] for p in common)
    only_low = sum(per[low][p] and not per[high][p] for p in common)
    only_high = sum(per[high][p] and not per[low][p] for p in common)
    rates = {}
    for label in (low, high):
        m = main[(main["model"] == label) & main["key"].isin(set(g.loc[g["model"] == label, "key"]))
                 & main["prompt_id"].isin(common)]
        rates[label] = {"mentions": int(len(m)), "ghosts": int(m["y_ghost"].sum()),
                        "ghost_rate": float(m["y_ghost"].mean()) if len(m) else float("nan")}
    return {"sample": sample, "n_prompts": len(common), "rates": rates,
            "prompts_with_ghost": {low: int(sum(per[low][p] for p in common)),
                                   high: int(sum(per[high][p] for p in common))},
            "both": int(both), "only_low": int(only_low), "only_high": int(only_high),
            "neither": int(len(common) - both - only_low - only_high)}


def local_only(labelled, gens):
    """Drop hosted rows: the detector is trained and evaluated on the local open models only."""
    gens = gens[gens["arm"] != "hosted"] if "arm" in gens else gens
    return labelled[labelled["key"].isin(set(gens["key"]))], gens
