"""The ``llm_selfcheck`` baseline (BUILD_SPEC §8 arm 6): ask a model whether a name exists.

Only the name is sent to the local model; nothing is installed or fetched. Answers are
cached per (model, name) under ``data/cache/selfcheck/`` (gitignored: names include ghosts).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from phantom import config, generate
from phantom.http import cache_key

MODEL = "gemma4:e4b"
PROMPT = "Does a package named `{name}` exist on PyPI? Answer only yes or no."
SEED = config.GLOBAL_SEED
TEMPERATURE = 0.0
NUM_PREDICT = 8
_ANSWER = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_answer(text: str) -> str:
    """``"yes"``, ``"no"`` or ``"unclear"`` from the first yes/no word in the reply."""
    m = _ANSWER.search(text or "")
    return m.group(1).lower() if m else "unclear"


def score(answer: str) -> float:
    """Ghost score: 1 if the model says the package does not exist, else 0 (§8)."""
    return 1.0 if answer == "no" else 0.0


def cache_dir(model: str = MODEL):
    """Per-model answer cache directory."""
    return config.CACHE / "selfcheck" / re.sub(r"[^\w.-]", "_", model)


def ask_all(names: Iterable[str], model: str = MODEL,
            progress: Callable[[int, int], None] | None = None) -> dict[str, dict[str, Any]]:
    """Answers for every name, calling Ollama only for names not cached yet."""
    names = sorted(set(names))
    folder = cache_dir(model)
    folder.mkdir(parents=True, exist_ok=True)
    host = config.get_settings().ollama_host
    tags = {m["tag"]: m for m in generate.ollama_tags(host)}
    thinking = "thinking" in tags[model]["capabilities"]
    out: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=120.0) as client:
        for i, name in enumerate(names, 1):
            path = folder / f"{cache_key(name)}.json"
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                r = generate.ollama_chat(client, host, model, PROMPT.format(name=name), SEED, thinking,
                                         temperature=TEMPERATURE, num_predict=NUM_PREDICT, system=None)
                rec = {"name": name, "model": model, "digest": tags[model]["digest"], "reply": r["output"],
                       "answer": parse_answer(r["output"]), "latency_s": r["latency_s"]}
                path.write_text(json.dumps(rec), encoding="utf-8")
                out[name] = rec
            if progress:
                progress(i, len(names))
        client.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0})  # free VRAM/RAM
    return out
