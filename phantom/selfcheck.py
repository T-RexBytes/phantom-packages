"""The ``llm_selfcheck`` baselines (BUILD_SPEC §8 arm 6): ask a model whether a name exists.

Only the name is sent to the local model; nothing is installed or fetched.
- ``llm_selfcheck`` (spec): score 1 if the reply says "no", else 0.
- ``llm_selfcheck_prob`` (fairer): P(no) / (P(no) + P(yes)) from the first answer token's
  top-k alternatives, so the model's confidence counts, not just its argmax.
Answers are cached per (model, name) under ``data/cache/selfcheck/`` (hard answers) and
``data/cache/selfcheck_v2/`` (with top logprobs); both gitignored, since names include ghosts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

from phantom import config, generate
from phantom.http import cache_key

MODEL = "gemma4:e4b"
PROMPT = "Does a package named `{name}` exist on PyPI? Answer only yes or no."
SEED = config.GLOBAL_SEED
TEMPERATURE = 0.0
NUM_PREDICT = 8
TOP_K = 20  # alternatives per position for the probabilistic score
_ANSWER = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_NOISE = " \t\r\n*`_\"'.:!,()[]"  # markup/whitespace stripped before reading an answer token


def parse_answer(text: str) -> str:
    """``"yes"``, ``"no"`` or ``"unclear"`` from the first yes/no word in the reply."""
    m = _ANSWER.search(text or "")
    return m.group(1).lower() if m else "unclear"


def score(answer: str) -> float:
    """Ghost score: 1 if the model says the package does not exist, else 0 (§8)."""
    return 1.0 if answer == "no" else 0.0


def p_no(positions: list | None) -> float | None:
    """P(no) / (P(no) + P(yes)) at the first position whose token isn't whitespace/markup.

    ``positions`` are ``[token, logprob, [[alternative, logprob], ...]]`` (``ollama_chat``
    with ``top_logprobs``). Alternatives are folded case- and space-insensitively ("No", " no").
    Returns None if neither answer is among that position's alternatives.
    """
    for token, _lp, alts in positions or []:
        if not (token or "").strip(_NOISE):
            continue
        mass = {"yes": 0.0, "no": 0.0}
        for alt, lp in alts or []:
            word = (alt or "").strip(_NOISE).lower()
            if word in mass and lp is not None:
                mass[word] += math.exp(lp)
        total = mass["yes"] + mass["no"]
        return mass["no"] / total if total > 0 else None
    return None


def prob_score(record: dict[str, Any]) -> float:
    """Probabilistic ghost score: ``p_no`` when available, else the hard answer's score."""
    return record["p_no"] if record.get("p_no") is not None else score(record["answer"])


def cache_dir(model: str = MODEL, with_logprobs: bool = False) -> Path:
    """Per-model answer cache directory (v2 = answers stored with top logprobs)."""
    return config.CACHE / ("selfcheck_v2" if with_logprobs else "selfcheck") / re.sub(r"[^\w.-]", "_", model)


def ask_all(names: Iterable[str], model: str = MODEL, progress: Callable[[int, int], None] | None = None,
            top_k: int | None = None) -> dict[str, dict[str, Any]]:
    """Answers for every name, calling Ollama only for names not cached yet.

    With ``top_k`` each record also has ``p_no`` and the first positions' alternatives.
    """
    names = sorted(set(names))
    folder = cache_dir(model, with_logprobs=bool(top_k))
    folder.mkdir(parents=True, exist_ok=True)
    paths = {n: folder / f"{cache_key(n)}.json" for n in names}
    out = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in paths.items() if p.exists()}
    missing = [n for n in names if n not in out]
    if not missing:  # fully cached: no Ollama needed (``tasks.py all --fast`` works with it off)
        if progress:
            progress(len(names), len(names))
        return out
    host = config.get_settings().ollama_host
    tags = {m["tag"]: m for m in generate.ollama_tags(host)}
    thinking = "thinking" in tags[model]["capabilities"]
    with httpx.Client(timeout=120.0) as client:
        for i, name in enumerate(missing, 1):
            r = generate.ollama_chat(client, host, model, PROMPT.format(name=name), SEED, thinking,
                                     temperature=TEMPERATURE, num_predict=NUM_PREDICT, system=None,
                                     top_logprobs=top_k)
            rec = {"name": name, "model": model, "digest": tags[model]["digest"], "reply": r["output"],
                   "answer": parse_answer(r["output"]), "latency_s": r["latency_s"]}
            if top_k:
                rec.update(top_k=top_k, p_no=p_no(r.get("top_logprobs")), top_logprobs=(r.get("top_logprobs") or [])[:3])
            paths[name].write_text(json.dumps(rec), encoding="utf-8")
            out[name] = rec
            if progress:
                progress(len(names) - len(missing) + i, len(names))
        client.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0})  # free VRAM/RAM
    return out
