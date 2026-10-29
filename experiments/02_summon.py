"""Stage 02: summon generations from local Ollama models (BUILD_SPEC §5.2).

Subcommands:
  tags                      write results/models.json from GET /api/tags
  smoke --models a,b,c      5 prompts per model -> latency, ETA (results/smoke.json)
  run   [--samples 2]       resumable full run -> data/raw_generations.jsonl
  status                    counts of finished generations (also used by --fast)

Raw outputs never go to results/: they can contain unregistered ghost names.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

import httpx
from rich.console import Console

from phantom import config, generate, prompts, runconfig

SMOKE_PATH = config.DATA / "smoke_generations.jsonl"
SMOKE_SUMMARY = config.RESULTS / "smoke.json"
MAX_FAILURES_IN_A_ROW = 3  # each already retried 3x; then assume Ollama is down
console = Console(highlight=False)


def write_models_json(selected: list[str] | None = None) -> dict:
    """Snapshot installed models (and the selection, if given) to results/models.json."""
    host = config.get_settings().ollama_host
    version = httpx.get(f"{host}/api/version", timeout=10).json().get("version")
    data = runconfig.load(config.MODELS_JSON_PATH) if config.MODELS_JSON_PATH.exists() else {}
    data.pop("cuts", None)
    data.update(
        {"checked_at": runconfig.utc_now(), "ollama_version": version,
         "installed": generate.ollama_tags(host)}
    )
    if selected is not None:
        data["selected"] = selected
    runconfig.save(data, config.MODELS_JSON_PATH)
    return data


def model_info(tags: list[str]) -> dict[str, dict]:
    """Installed-model metadata for ``tags``; exits if any tag is missing."""
    installed = {m["tag"]: m for m in write_models_json()["installed"]}
    missing = [t for t in tags if t not in installed]
    if missing:
        sys.exit(f"not installed in Ollama: {missing}")
    return {t: installed[t] for t in tags}


def smoke_prompt_ids(rows: list[dict]) -> list[str]:
    """Five fixed prompts mirroring the level mix: 1 x L1, 2 x L2, 2 x L3."""
    rng = random.Random(config.GLOBAL_SEED)
    pick: list[str] = []
    for level, n in ((1, 1), (2, 2), (3, 2)):
        pick += rng.sample([r["id"] for r in rows if r["niche_level"] == level], n)
    return pick


def call_with_retry(client, host, model, info, text, seed, tries=3) -> dict | None:
    """One generation with a few retries on HTTP/transport errors; None if all fail."""
    for attempt in range(tries):
        try:
            return generate.ollama_chat(
                client, host, model, text, seed, "thinking" in info["capabilities"]
            )
        except httpx.HTTPError as exc:
            console.print(f"[yellow]  {model}: {type(exc).__name__}: {exc} (try {attempt + 1}/{tries})")
            time.sleep(5 * (attempt + 1))
    return None


def execute(models: list[str], prompt_ids: list[str], samples: list[int], path) -> Counter:
    """Run the plan, skipping finished keys; append each result with fsync."""
    host = config.get_settings().ollama_host
    info = model_info(models)
    by_id = {r["id"]: r for r in prompts.load_prompts()}
    done = generate.done_keys(path)
    todo = [
        (s, m, p) for s, m, p in generate.plan(models, prompt_ids, samples)
        if generate.generation_key(m, p, s) not in done
    ]
    total = len(todo)
    console.print(f"{len(done)} already done, {total} to go -> {path.relative_to(config.ROOT)}")
    eta, failures, in_a_row = generate.Eta(), Counter(), 0
    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        for i, (sample, model, pid) in enumerate(todo, 1):
            seed = config.SAMPLE_SEED_BASE + sample
            result = call_with_retry(client, host, model, info[model], by_id[pid]["text"], seed)
            if result is None:
                failures[model] += 1
                in_a_row += 1
                if in_a_row >= MAX_FAILURES_IN_A_ROW:
                    console.print(f"[red]{in_a_row} calls failed in a row (is Ollama running?); "
                                  "stopping. Re-run the same command to resume.")
                    break
                continue
            in_a_row = 0
            generate.append_record(
                path, generate.local_record(model, info[model]["digest"], pid, sample, seed, result)
            )
            eta.add(model, result["latency_s"] - result["load_duration_s"])
            left = eta.remaining_s(m for _, m, _ in todo[i:])
            console.print(
                f"[{i}/{total}] s{sample} {model} {pid} {result['latency_s']:.1f}s "
                f"tok={result['eval_count']} {result['done_reason']} | ETA {left / 3600:.2f}h",
            )
    if failures:
        console.print(f"[red]failed (will retry on resume): {dict(failures)}")
    return failures


def cmd_smoke(args) -> int:
    models = args.models.split(",")
    rows = prompts.load_prompts()
    ids = smoke_prompt_ids(rows)
    execute(models, ids, [1], SMOKE_PATH)
    recs = [r for r in generate.read_records(SMOKE_PATH) if r["model"] in models and r["prompt_id"] in ids]
    per_model = {}
    for m in models:
        mr = [r for r in recs if r["model"] == m]
        steady = [r["latency_s"] - r["load_duration_s"] for r in mr]
        per_model[m] = {
            "n": len(mr),
            "mean_latency_s": round(sum(steady) / len(steady), 2),
            "max_load_s": round(max(r["load_duration_s"] for r in mr), 2),
            "mean_eval_count": round(sum(r["eval_count"] or 0 for r in mr) / len(mr), 1),
            "thinking_chars_total": sum(r["thinking_chars"] for r in mr),
            "done_reasons": dict(Counter(r["done_reason"] for r in mr)),
            "logprobs_returned": all(r["logprobs"] for r in mr),
        }
    per_call = sum(v["mean_latency_s"] for v in per_model.values()) / len(per_model)
    n = len(rows)
    eta_h = {f"{k}_samples": round(per_call * len(models) * n * k / 3600, 2) for k in (1, 2, 3)}
    summary = {"checked_at": runconfig.utc_now(), "prompt_ids": ids, "per_model": per_model,
               "n_prompts": n, "eta_hours": eta_h, "budget_hours": config.ETA_BUDGET_HOURS}
    runconfig.save(summary, SMOKE_SUMMARY)

    for m in models:
        r = next(r for r in recs if r["model"] == m)
        console.rule(f"{m} | {r['prompt_id']} | {r['latency_s']:.1f}s | {r['eval_count']} tokens")
        console.print(r["output"], markup=False)
    console.rule("smoke summary")
    for m, v in per_model.items():
        console.print(f"{m}: {v}")
    console.print(f"ETA full run ({len(models)} models x {n} prompts): {eta_h} h "
                  f"(budget {config.ETA_BUDGET_HOURS} h)")
    return 0


def cmd_run(args) -> int:
    models = args.models.split(",") if args.models else write_models_json().get("selected")
    if not models:
        sys.exit("no models: pass --models or run `tags --select ...` first")
    rows = prompts.load_prompts()
    ids = prompts.pilot_ids(rows, args.pilot) if args.pilot else [r["id"] for r in rows]
    ids = ids[: args.prompt_limit or None]
    samples = list(range(1, args.samples + 1))
    runconfig.update("summon", {"models": models, "samples": samples, "n_prompts": len(ids),
                                "pilot": args.pilot or None,
                                "temperature": config.TEMPERATURE, "num_predict": config.NUM_PREDICT,
                                "num_ctx": config.NUM_CTX, "think": False,
                                "started_at": runconfig.utc_now()})
    failures = execute(models, ids, samples, config.RAW_GENERATIONS_PATH)
    return 1 if failures else 0


def cmd_hosted(args) -> int:
    """Ultramarine arm, sample 1. Refuses to spend without --confirm-spend."""
    from phantom import hosted

    key = config.get_settings().paperinstruments_api_key
    if not key:
        sys.exit("PAPERINSTRUMENTS_API_KEY not set: Ultramarine arm skipped")
    rows = prompts.load_prompts()[: args.prompt_limit or None]
    done = generate.done_keys(config.RAW_GENERATIONS_PATH)
    todo = [r for r in rows if generate.generation_key(hosted.MODEL, r["id"], 1) not in done]
    card = hosted.fetch_pricing()
    bound = hosted.max_cost_usd(len(todo), 120, card)
    console.print(f"{len(todo)} hosted calls; worst-case spend ${bound:.4f} "
                  f"(rates saved to {hosted.PRICING_PATH.relative_to(config.ROOT)})")
    if not args.confirm_spend:
        console.print("dry run: re-run with --confirm-spend after the user approves")
        return 0
    from openai import OpenAI

    client = OpenAI(base_url=hosted.BASE_URL, api_key=key)
    spent = 0.0
    for i, r in enumerate(todo, 1):
        rec = hosted.ultramarine_record(client, r["id"], r["text"], card)
        generate.append_record(config.RAW_GENERATIONS_PATH, rec)
        spent += rec["cost_usd"]
        console.print(f"[{i}/{len(todo)}] {r['id']} {rec['latency_s']:.1f}s "
                      f"tok={rec['eval_count']} {rec['done_reason']} | spent ${spent:.4f}")
    return 0


def cmd_status(_args) -> int:
    recs = list(generate.read_records(config.RAW_GENERATIONS_PATH))
    counts = Counter((r["model"], r["sample"]) for r in recs)
    console.print(f"{len(recs)} generations in {config.RAW_GENERATIONS_PATH.relative_to(config.ROOT)}")
    for (m, s), c in sorted(counts.items()):
        console.print(f"  {m} s{s}: {c}")
    return 0


def cmd_tags(args) -> int:
    data = write_models_json(args.select.split(",") if args.select else None)
    for m in data["installed"]:
        console.print(f"{m['tag']:<18} {m['family']:<8} {m['parameter_size']:<6} {m['quantization']} "
                      f"{m['digest'][:12]} {m['capabilities']}")
    console.print(f"selected: {data.get('selected')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="no model calls: print status only")
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("tags")
    p.add_argument("--select", help="comma-separated tags to record as the study models")
    p = sub.add_parser("smoke")
    p.add_argument("--models", required=True)
    p = sub.add_parser("run")
    p.add_argument("--models")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--prompt-limit", type=int, default=0, help="first N prompts only (scope cut)")
    p.add_argument("--pilot", type=int, default=0, help="balanced subset of N prompts (prompts.pilot_ids)")
    p = sub.add_parser("hosted", help="Ultramarine arm (sample 1); dry run unless --confirm-spend")
    p.add_argument("--confirm-spend", action="store_true")
    p.add_argument("--prompt-limit", type=int, default=0)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.fast or args.cmd in (None, "status"):
        return cmd_status(args)
    return {"tags": cmd_tags, "smoke": cmd_smoke, "run": cmd_run, "hosted": cmd_hosted}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
