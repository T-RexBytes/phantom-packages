"""Ultramarine hosted arm (BUILD_SPEC §5.2): same prompts, sample 1 only.

Contract verified 2026-10-28 (docs/api_notes/tabpfn_ultramarine.md): OpenAI-
compatible chat completions at ``https://api.paperinstruments.com/v1``;
``reasoning_effort`` is ``low|high|max`` (default ``max``, so we send ``low``);
``max_completion_tokens`` includes reasoning tokens; ``GET /v1/pricing`` is
public and returns integer nanodollar rates. Billing is prepaid.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from phantom import config, runconfig
from phantom.generate import generation_key, strip_thinking
from phantom.runconfig import utc_now

BASE_URL = "https://api.paperinstruments.com/v1"
MODEL = "ultramarine-1-flash"
REASONING_EFFORT = "low"
# Reasoning tokens count against this cap; 350 visible tokens + reasoning headroom.
MAX_COMPLETION_TOKENS = 1024
PRICING_PATH = config.RESULTS / "ultramarine_pricing.json"


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


def max_cost_usd(n_calls: int, prompt_tokens_each: int, card: dict) -> float:
    """Upper bound on spend if every call used the full completion budget."""
    return n_calls * cost_usd(prompt_tokens_each, 0, MAX_COMPLETION_TOKENS, card)


def ultramarine_record(client: Any, prompt_id: str, text: str, card: dict) -> dict[str, Any]:
    """One hosted generation as a ``raw_generations.jsonl`` row (``arm="hosted"``)."""
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=config.TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        seed=config.SAMPLE_SEED_BASE + 1,
    )
    latency = time.perf_counter() - start
    u = resp.usage
    cached = (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0
    reasoning = (u.completion_tokens_details.reasoning_tokens if u.completion_tokens_details else 0) or 0
    choice = resp.choices[0]
    return {
        "key": generation_key(MODEL, prompt_id, 1),
        "model": MODEL,
        "digest": None,
        "prompt_id": prompt_id,
        "sample": 1,
        "seed": config.SAMPLE_SEED_BASE + 1,
        "temperature": config.TEMPERATURE,
        "output": strip_thinking(choice.message.content or ""),
        "done_reason": choice.finish_reason,
        "latency_s": round(latency, 3),
        "eval_count": u.completion_tokens,
        "prompt_eval_count": u.prompt_tokens,
        "usage": {
            "prompt_tokens": u.prompt_tokens,
            "cached_tokens": cached,
            "completion_tokens": u.completion_tokens,
            "reasoning_tokens": reasoning,
        },
        "cost_usd": cost_usd(u.prompt_tokens, cached, u.completion_tokens, card),
        "reasoning_effort": REASONING_EFFORT,
        "ts": utc_now(),
        "arm": "hosted",
    }
