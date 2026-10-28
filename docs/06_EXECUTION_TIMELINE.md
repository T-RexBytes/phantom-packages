# 06: Execution Timeline

**Hard stop:** Fri **Oct 30, 11:59 PM PDT**. That's **Sat Oct 31, 12:29 PM IST** and **Oct 31, 06:59 UTC**.

**Internal publish target:** **Oct 30, 8:00 PM PDT** (Oct 31, 8:30 AM IST). That leaves 4 hours of buffer.

Times below are relative blocks. Map them to your own day. "H" means hour of work.

---

## Day 1 (Wed Oct 28 to Thu Oct 29 morning): build the dataset

| Block | Task | Output | Cut line if late |
|---|---|---|---|
| H0–0.5 | Setup: Ollama models pulled, uv, TabPFN license + token, promos claimed (Ultramarine $5, .tech), GitHub repo | env ready | skip Ultramarine and .tech |
| H0.5–1.5 | **Phase 0** scaffold (Claude Code) | repo, CI, tests skeleton | — |
| H1.5–2.5 | **Phase 1** prompts | `prompts.jsonl` (~200) | 120 prompts |
| H2.5–3 | **Phase 2** smoke test + ETA | `models.json`, ETA | drop to 2 models if ETA over 6h |
| H3 → overnight | **Phase 2** full summon runs **in the background**. Keep the laptop plugged in, sleep disabled. | `raw_generations.jsonl` | stop at a complete sample-1 sweep |
| H3–5 (in parallel) | **Phase 3** extractor (TDD) + registry on partial data | tests green | — |

**Before sleeping:** confirm the run is resumable. Kill and restart it once to prove it.

## Day 2 (Thu Oct 29): results

| Block | Task | Output | Cut line |
|---|---|---|---|
| H0–1 | Finish Phase 3 on full data | mentions + registry parquet | — |
| H1–2 | **Phase 4** features + leakage test | `dataset.parquet` | drop `char_lp` and `name_logprob` |
| H2–4 | **Phase 5** eval (6 arms × 3 splits) | `eval.csv`, `loo.csv`, calibration, `summary.json` | drop the llm_selfcheck arm; split C optional |
| H4–5 | **Phase 6** squat check | `squat_public.csv`, private files | — |
| H5–7 | **Phase 7** guard CLI + hook + Action | working tool, tests | Action optional |
| H7–7.5 | **Record the hook demo** (see 07) | `demo.gif` | screenshot instead of GIF |
| H7.5–9 | **Phase 8** figures + site + deploy | live site | plain page with tables only |
| Evening | Real-person moment: send the hook to 1–2 friends who use AI agents | reaction screenshot | your own session moment |

## Day 3 (Fri Oct 30): write and ship (writing is weighted most, so protect this day)

| Block (PDT) | Task | Output |
|---|---|---|
| Morning | **Phase 9** POST_DRAFT generated from results | draft |
| +2h | **You** rewrite the hook, What I Built, Spirit of It and the "why open" section in your own voice | v2 |
| +1h | Judge self-review prompt (04 recovery) + the checklist in 08 | v3 |
| +1h | Media: cover image, GIF upload, alt texts, Mermaid check in DEV preview | v4 |
| +0.5h | DevRelay: save + embed the agent session | v5 |
| **by 8 PM PDT** | **Publish** with tags `devchallenge, hf26challenge, python, security` | live post |
| After | Share in the MLH Discord / socials; answer comments on the post | engagement |

---

## If you're behind: the fallback ladder

Cut from the top first; keep the core story.

1. Drop the Ultramarine arm (lose the $100 category entry).
2. Drop the GitHub Action (keep CLI + hook).
3. Drop split C and the llm_selfcheck arm.
4. Drop to 2 local models (keep leave-one-model-out as 2-way: train A, test B, and the reverse).
5. Drop the site and use GitHub README + images instead. You lose .tech, but the demo stays.
6. **Never drop:** the eval table with baselines, What Went Wrong, the hook demo, the template headings, the honest "why open".

## Daily stand-up prompt for Claude Code
```text
Status check: read results/run_config.json and git log. Hours left until Oct 30 8 PM PDT? Which phases are done? What's the single most important next step? Anything we should cut now to protect the writing time on Day 3?
```
