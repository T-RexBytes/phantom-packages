"""Tests for the Ollama caller, resumable log and ETA (mocked HTTP, no models)."""

from __future__ import annotations

import json

import httpx
import pytest

from phantom import config, generate


def chat_client(captured, payload):
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


PAYLOAD = {
    "message": {"role": "assistant", "content": "<think>hmm</think>```bash\npip install requests\n```"},
    "done_reason": "stop",
    "total_duration": 2_500_000_000,
    "load_duration": 500_000_000,
    "eval_count": 12,
    "prompt_eval_count": 40,
    "logprobs": [{"token": "pip", "logprob": -0.1, "bytes": [1]}],
}


def test_chat_request_body_and_parsing():
    captured = []
    with chat_client(captured, PAYLOAD) as client:
        out = generate.ollama_chat(client, "http://h", "m:1", "do x", seed=1001, supports_thinking=True)
    body = captured[0]
    assert body["think"] is False and body["stream"] is False
    assert body["options"] == {"temperature": 0.7, "seed": 1001, "num_predict": 350, "num_ctx": 1024}
    assert body["messages"][0] == {"role": "system", "content": config.SYSTEM_PROMPT}
    assert out["output"] == "```bash\npip install requests\n```"
    assert out["thinking_chars"] == len("<think>hmm</think>")
    assert out["total_duration_s"] == 2.5 and out["load_duration_s"] == 0.5
    assert out["logprobs"] == [["pip", -0.1]]


def test_no_think_flag_for_models_without_thinking():
    captured = []
    with chat_client(captured, {"message": {"content": "x"}}) as client:
        out = generate.ollama_chat(client, "http://h", "m", "p", 1, supports_thinking=False)
    assert "think" not in captured[0]
    assert out["logprobs"] is None and out["thinking_chars"] == 0


def test_plan_order_is_samples_then_models_then_prompts():
    order = generate.plan(["a", "b"], ["p1", "p2"], [1, 2])
    assert order[:4] == [(1, "a", "p1"), (1, "a", "p2"), (1, "b", "p1"), (1, "b", "p2")]
    assert order[4] == (2, "a", "p1") and len(order) == 8


def test_append_and_resume_tolerates_torn_line(tmp_path):
    path = tmp_path / "raw.jsonl"
    rec = generate.local_record("m", "sha256:x", "p001", 1, 1001, {"output": "o"})
    assert rec["key"] == "m|p001|s1" and rec["arm"] == "local"
    generate.append_record(path, rec)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "m|p002|s1", "outp')  # crash mid-write
    assert generate.done_keys(path) == {"m|p001|s1"}


@pytest.mark.parametrize("text,expected", [("a <THINK>x\ny</think>\nb", "a b"), ("plain", "plain")])
def test_strip_thinking(text, expected):
    assert generate.strip_thinking(text) == expected


def test_eta_uses_per_model_means_and_fallback():
    eta = generate.Eta(window=2)
    for lat in (10.0, 2.0, 4.0):  # window keeps last two -> mean 3
        eta.add("a", lat)
    eta.add("b", 7.0)
    assert eta.mean("a") == 3.0
    assert eta.remaining_s(["a", "b", "c"]) == 3.0 + 7.0 + 5.0
