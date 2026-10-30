"""Tests for Ultramarine cost accounting (no network)."""

from __future__ import annotations

import pytest

from phantom import hosted

# Price card shape as fetched from GET /v1/pricing on 2026-10-28.
CARD = {
    "uncached_input_nanodollars_per_token": 150,
    "cached_input_nanodollars_per_token": 30,
    "output_nanodollars_per_token": 500,
}


def test_cost_uses_nanodollar_rates():
    # (100-20)*150 + 20*30 + 300*500 = 12_000 + 600 + 150_000 nanodollars
    assert hosted.cost_usd(100, 20, 300, CARD) == pytest.approx(162_600 / 1e9)


def test_max_cost_bound_for_full_run_is_small():
    bound = hosted.max_cost_usd(200, 120, CARD)
    assert bound == pytest.approx(200 * (120 * 150 + hosted.MAX_COMPLETION_TOKENS * 500) / 1e9)
    assert bound < 0.15


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def fake_response(content="```bash\npip install requests\n```", prompt=90, cached=10, completion=300,
                  reasoning=120, logprobs=None, finish="stop"):
    usage = _Obj(prompt_tokens=prompt, completion_tokens=completion,
                 prompt_tokens_details=_Obj(cached_tokens=cached),
                 completion_tokens_details=_Obj(reasoning_tokens=reasoning))
    choice = _Obj(message=_Obj(content=content), finish_reason=finish, logprobs=logprobs)
    return _Obj(usage=usage, choices=[choice])


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_ledger_caps_and_persists(tmp_path):
    path = tmp_path / "spend.json"
    led = hosted.Ledger(path, cap_usd=0.01)
    led.check(0.005)
    led.add("A_low", 0.006, {"prompt_tokens": 90, "completion_tokens": 300, "reasoning_tokens": 100})
    with pytest.raises(hosted.BudgetExceeded):
        led.check(0.005)  # 0.006 + 0.005 > 0.01
    again = hosted.Ledger(path, cap_usd=0.01)  # a restart sees the spend
    assert again.spent == pytest.approx(0.006) and again.data["parts"]["A_low"]["calls"] == 1


def test_record_from_fake_response_uses_label_sample_and_cost():
    client = FakeClient([fake_response()])
    rec = hosted.ultramarine_record(client, "p001", "task", CARD, sample=2, effort="high",
                                    max_completion=4096, label=hosted.MODEL_HIGH)
    kw = client.calls[0]
    assert kw["model"] == hosted.MODEL and kw["reasoning_effort"] == "high" and kw["max_completion_tokens"] == 4096
    assert kw["seed"] == 1002 and rec["seed"] == 1002
    assert rec["key"] == f"{hosted.MODEL_HIGH}|p001|s2" and rec["model"] == hosted.MODEL_HIGH
    assert rec["arm"] == "hosted" and rec["usage"]["reasoning_tokens"] == 120
    assert rec["cost_usd"] == pytest.approx(((90 - 10) * 150 + 10 * 30 + 300 * 500) / 1e9)


def test_retry_honours_retry_after_then_succeeds():
    import httpx
    import openai

    req = httpx.Request("POST", "https://x/v1/chat/completions")
    err = openai.RateLimitError("slow down", response=httpx.Response(429, headers={"retry-after": "7"}, request=req),
                                body=None)
    waits = []
    client = FakeClient([err, fake_response()])
    rec = hosted.ultramarine_record(client, "p001", "task", CARD, sleep=waits.append)
    assert waits == [7.0] and rec["output"].startswith("```bash")


def test_selfcheck_record_reads_p_no_from_logprobs():
    top = [_Obj(token="No", logprob=-0.1), _Obj(token="Yes", logprob=-2.4)]
    lp = _Obj(content=[_Obj(token="No", logprob=-0.1, top_logprobs=top)])
    client = FakeClient([fake_response(content="No", logprobs=lp)])
    rec = hosted.selfcheck_record(client, "zzq-ghostlib", CARD)
    assert rec["answer"] == "no" and 0.9 < rec["p_no"] < 1.0
    assert client.calls[0]["top_logprobs"] == hosted.SELFCHECK_TOP_LOGPROBS


def test_selfcheck_skips_the_think_close_token():
    think_alts = [_Obj(token="</think>", logprob=-0.4), _Obj(token="Yes", logprob=-1.3)]
    ans_alts = [_Obj(token="No", logprob=-0.05), _Obj(token="Yes", logprob=-3.0)]
    lp = _Obj(content=[_Obj(token="</think>", logprob=-0.4, top_logprobs=think_alts),
                       _Obj(token="No", logprob=-0.05, top_logprobs=ans_alts)])
    rec = hosted.selfcheck_record(FakeClient([fake_response(content="No", logprobs=lp)]), "zzq-ghostlib", CARD)
    assert rec["p_no"] > 0.9  # read after </think>, not at it


def test_hosted_in_plan_keeps_hosted_rows_for_planned_prompts_only():
    recs = [{"arm": "hosted", "prompt_id": "p001"}, {"arm": "hosted", "prompt_id": "p999"},
            {"arm": "local", "prompt_id": "p001"}, {"prompt_id": "p001"}]
    assert hosted.hosted_in_plan(recs, ["p001"]) == [{"arm": "hosted", "prompt_id": "p001"}]


def _hosted_frames():
    import pandas as pd

    gens = pd.DataFrame({
        "key": ["L|p1|s1", "u|p1|s1", "u|p2|s1", "h|p1|s1", "h|p2|s1"],
        "model": ["gemma", "u", "u", "h", "h"], "prompt_id": ["p1", "p1", "p2", "p1", "p2"],
        "sample": [1] * 5, "arm": ["local", "hosted", "hosted", "hosted", "hosted"]})
    lab = pd.DataFrame({
        "key": ["L|p1|s1", "u|p1|s1", "u|p2|s1", "h|p1|s1", "h|p2|s1", "u|p2|s1"],
        "model": ["gemma", "u", "u", "h", "h", "u"], "prompt_id": ["p1", "p1", "p2", "p1", "p2", "p2"],
        "name": ["requests", "zzq-a", "numpy", "requests", "numpy", "json"],
        "category": ["real", "ghost", "real", "real", "real", "stdlib_confusion"],
        "y_ghost": [False, True, False, False, False, False]})
    return lab, gens


def test_local_only_drops_hosted_rows_from_the_detector_inputs():
    lab, gens = _hosted_frames()
    lab_l, gens_l = hosted.local_only(lab, gens)
    assert set(gens_l["arm"]) == {"local"} and set(lab_l["model"]) == {"gemma"}


def test_ablation_counts_discordant_prompts():
    lab, gens = _hosted_frames()
    ab = hosted.ablation(lab, gens, low="u", high="h")
    assert ab["n_prompts"] == 2 and ab["only_low"] == 1 and ab["only_high"] == 0 and ab["neither"] == 1
    assert ab["rates"]["u"] == {"mentions": 2, "ghosts": 1, "ghost_rate": 0.5}  # stdlib excluded


def test_hosted_stats_tokens_and_cost():
    lab, gens = _hosted_frames()
    recs = [{"model": m, "usage": {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 5},
             "cost_usd": 0.001, "reasoning_effort": e, "done_reason": "stop"}
            for m, e in (("u", "low"), ("u", "low"), ("h", "high"), ("h", "high"))]
    st = hosted.hosted_stats(lab[lab["model"] != "gemma"], gens[gens["arm"] == "hosted"], recs)
    assert st["u"]["ghosts"] == 1 and st["u"]["gen_share"] == 0.5 and st["u"]["reasoning_tokens"] == 10
    assert st["h"]["cost_per_1k_generations_usd"] == pytest.approx(1.0)
