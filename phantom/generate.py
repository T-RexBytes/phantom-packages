"""Model callers and the resumable generation log (BUILD_SPEC §4.2, §5.2).

Ollama contract verified 2026-10-28 (docs/api_notes/ollama_pypi_hooks.md):
``POST /api/chat`` with ``stream: false``; ``think: false`` disables thinking on
models whose ``/api/tags`` capabilities include ``"thinking"``; durations are
nanoseconds; ``logprobs`` is a top-level array of ``{token, logprob, ...}``.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from phantom import config
from phantom.runconfig import utc_now

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def generation_key(model: str, prompt_id: str, sample: int) -> str:
    """Unique key of one generation, e.g. ``gemma4:e4b|p042|s1``."""
    return f"{model}|{prompt_id}|s{sample}"


def ollama_tags(host: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Return installed models from ``GET /api/tags`` in the ``models.json`` shape."""
    models = httpx.get(f"{host}/api/tags", timeout=timeout).raise_for_status().json()["models"]
    return [
        {
            "tag": m["name"],
            "digest": m.get("digest"),
            "size_bytes": m.get("size"),
            "family": m.get("details", {}).get("family"),
            "parameter_size": m.get("details", {}).get("parameter_size"),
            "quantization": m.get("details", {}).get("quantization_level"),
            "capabilities": m.get("capabilities", []),
            "modified_at": m.get("modified_at"),
        }
        for m in models
    ]


def strip_thinking(text: str) -> str:
    """Remove any ``<think>…</think>`` blocks a model leaks into its answer."""
    return _THINK_BLOCK.sub("", text)


def ollama_chat(
    client: httpx.Client,
    host: str,
    model: str,
    prompt: str,
    seed: int,
    supports_thinking: bool,
    temperature: float = config.TEMPERATURE,
    num_predict: int = config.NUM_PREDICT,
    num_ctx: int = config.NUM_CTX,
) -> dict[str, Any]:
    """Call ``/api/chat`` once (non-streaming) and return the generation fields."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx,
        },
        "logprobs": True,
    }
    if supports_thinking:
        body["think"] = False
    start = time.perf_counter()
    data = client.post(f"{host}/api/chat", json=body).raise_for_status().json()
    latency = time.perf_counter() - start
    message = data.get("message", {})
    content = message.get("content", "")
    output = strip_thinking(content)
    logprobs = data.get("logprobs")
    return {
        "output": output,
        "thinking_chars": len(message.get("thinking") or "") + len(content) - len(output),
        "done_reason": data.get("done_reason"),
        "latency_s": round(latency, 3),
        "load_duration_s": round(data.get("load_duration", 0) / 1e9, 3),
        "total_duration_s": round(data.get("total_duration", 0) / 1e9, 3),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "logprobs": [[t.get("token"), t.get("logprob")] for t in logprobs] if logprobs else None,
    }


def local_record(
    model: str, digest: str | None, prompt_id: str, sample: int, seed: int, result: dict[str, Any]
) -> dict[str, Any]:
    """Assemble one ``raw_generations.jsonl`` row for the local (Ollama) arm."""
    return {
        "key": generation_key(model, prompt_id, sample),
        "model": model,
        "digest": digest,
        "prompt_id": prompt_id,
        "sample": sample,
        "seed": seed,
        "temperature": config.TEMPERATURE,
        **result,
        "ts": utc_now(),
        "arm": "local",
    }


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL log, skipping a torn last line from a crash."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def done_keys(path: Path) -> set[str]:
    """Keys already present in a generation log."""
    return {r["key"] for r in read_records(path)}


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line and ``fsync`` so a crash never loses a finished call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def plan(models: list[str], prompt_ids: list[str], samples: Iterable[int]) -> list[tuple[int, str, str]]:
    """Work order: samples outermost, then models, then prompts (balanced partial runs)."""
    return [(s, m, p) for s in samples for m in models for p in prompt_ids]


class Eta:
    """Rolling per-model latency average used to estimate remaining time."""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._lat: dict[str, list[float]] = {}

    def add(self, model: str, latency_s: float) -> None:
        """Record one call's latency."""
        self._lat.setdefault(model, []).append(latency_s)
        del self._lat[model][: -self.window]

    def mean(self, model: str) -> float | None:
        """Rolling mean latency for ``model`` (None before the first call)."""
        lat = self._lat.get(model)
        return sum(lat) / len(lat) if lat else None

    def remaining_s(self, todo: Iterable[str]) -> float:
        """Estimated seconds for the remaining calls (models listed once per call)."""
        known = [m for m in self._lat]
        fallback = sum(self.mean(m) for m in known) / len(known) if known else 0.0
        return sum(self.mean(m) or fallback for m in todo)
