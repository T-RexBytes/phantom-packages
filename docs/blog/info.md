# Blog notes: work in progress

Running notes for the post (`docs/05_BLOG_POST_BLUEPRINT.md`). Facts only, each with its source.
Numbers here are working copies; the post must quote `results/summary.json` / `results/*.json`.
**No unregistered ghost names in this file** (it is public). Ghost names live in `private/` only.

Last updated: 2026-10-29 (after the full generation run, before Phase 5).

---

## 1. Setup (for "How I Built It")

| What | Value | Source |
|---|---|---|
| Hardware | Laptop: RTX 3050 4 GB VRAM, Ryzen 7 7445HS, 15 GB RAM, Windows 11 | `results/run_config.json` → `env` |
| Runtime | Ollama 0.40.11, models stored on a second drive | `run_config.json` → `env.ollama` |
| Models | `gemma4:e4b` (Google), `qwen3:4b-instruct` (Alibaba), `phi4-mini:3.8b` (Microsoft), all Q4_K_M | `results/models.json` |
| Prompts | 200 tasks, 15 domains, niche levels L1/L2/L3 = 50/70/80 (150 seed tasks + 50 hand-written variants) | `run_config.json` → `prompts` |
| Prompt template | "Write a short Python script that {task}. Begin with the exact `pip install` command it needs." | `phantom/config.py` |
| System prompt | "You are a helpful Python developer. Reply with one bash code block containing the pip install command, then one python code block." | `phantom/config.py` |
| Sampling | temperature 0.7, 2 samples per prompt (seeds 1001, 1002), max 350 output tokens, context 1,024 | `run_config.json` → `summon` |
| Generations | 1,199 of 1,200 planned (1 aborted, see §3) | `results/label_summary.json` |
| Generation time (sum of call latencies) | gemma4:e4b 2.08 h, qwen3:4b-instruct 1.04 h, phi4-mini 0.51 h → about 3.6 h total | computed from `data/raw_generations.jsonl` |
| Median seconds per reply | gemma4 18.9 s, qwen3 9.5 s, phi4-mini 4.1 s | same |
| Replies cut by the 350-token cap | gemma4 80%, qwen3 30%, phi4-mini 2% | same (`done_reason == "length"`) |
| Registry checks | 447 unique names, metadata GETs only (PyPI JSON + pypistats), cached, ≤5 req/s | stage 03 log |

Rule kept throughout: nothing a model suggested was ever installed, downloaded or imported, even during the study.

---

## 2. Headline findings so far (full run, 1,199 generations)

From `results/label_summary.json` (subtypes excluded from the ghost rate):

| Model | Install suggestions | Ghosts | Ghost rate | Replies with ≥1 ghost |
|---|---|---|---|---|
| gemma4:e4b | 483 | 39 | 8.1% | 9.8% |
| phi4-mini:3.8b | 467 | 78 | 16.7% | 19.5% |
| qwen3:4b-instruct | 460 | 96 | 20.9% | 24.0% |
| **All** | **1,410** | **213** | **15.1%** | **17.8%** |

- **Niche matters a lot.** Ghost rate by niche level: L1 (everyday) 2.0% (7/356), L2 8.8% (46/521), L3 (obscure) 30.0% (160/533). L3 is about 15× L1.
- **162 unique ghost names.** 40 of them (24.7%) were suggested ≥2 times; 10 (6.2%) by ≥2 different models. Those are the squattable ones.
- **A ghost usually comes alone.** In 205 of the 213 replies that contained a ghost, the ghost was the *only* package in the `pip install` line. When a model invents, it doesn't pad a real list, it replaces the whole answer.
- **All 213 ghost mentions came from `pip install` lines**, none from `import` statements.
- **Subtypes (real mistakes, but not ghosts), counted separately:** 29 stdlib confusions (11 names, the `pip install json` kind), 7 import-name confusions (3 names, the `pip install cv2` kind). Use the spec's generic examples in the post, not names from our data.
- **"Born after summoning":** 12 mentions (7 names) are registered on PyPI today but were first published *after* the model that suggested them was released. Candidates for the Phase 6 squat check. Don't name them publicly until Phase 6 has looked at them.
- **Pilot vs full:** the 25% pilot (300 generations) gave 10.8%; the full run 15.1%. Small samples underestimate; say so in the post.
- **Dataset for the detector:** 888 rows (model, prompt, name), 173 ghosts (19.5%).
- **What separates ghosts from real names (means, real vs ghost):** edit distance to nearest top-15k name 0.32 vs 4.27; models suggesting it for the same prompt 2.01 vs 1.14; times seen across the run 23.1 vs 1.4; model's own token logprob −0.44 vs −0.74; name length 8.0 vs 11.4. No ghost is in the top-15k list (89% of real names are).

---

## 3. What went wrong (candidates for "What Went Wrong", all real)

1. **The "thinking" model that never answered.** `qwen3:4b` from Ollama is the thinking-only 2507 build. Even with `think: false` it reasoned in plain text until it hit the token cap: 0 of 5 smoke-test replies contained a `pip` command. Replaced with `qwen3:4b-instruct`. (`run_config.json` → `cuts`)
2. **Three labelling bugs found in real data, not in tests** (pilot, 300 generations): ghost mentions 41 → 39, ghost rate 11.5% → 10.8%, module-name confusions 8 → 2. (`run_config.json` → `label_fixes`)
   - A reply cut off by the token cap mid-name produced a fake ghost (half a name).
   - The import key `google.auth` normalized to the real distribution `google-auth` and was wrongly tagged as a module-name confusion.
   - A dotted stdlib path in a pip line (`xml.etree.ElementTree`) was counted as a ghost instead of a stdlib confusion.
3. **4 GB of VRAM was the real bottleneck.** With Ollama's default context, phi4-mini ran 36% on CPU. Shrinking the context to 1,024 tokens (prompts are ~90 tokens, replies ≤350) kept models on the GPU: qwen3 21 → 30 tok/s, phi4-mini 25 → 35 tok/s. (`phantom/config.py` comment)
4. **gemma4:e4b talks too much.** 80% of its replies hit the 350-token cap. It still gave the pip line first (the template asks for that), so labels are unaffected, but the extractor had to drop the unfinished last line of truncated replies.
5. **The first full run died at 73 generations** when Ollama went offline. The runner now stops after 3 consecutive failures instead of burning through the plan, and resumes by key (resume test: killed after 4, resumed with 0 duplicate keys). (`run_config.json` → `summon.runs`, `resume_test`)
6. **One generation can't be generated.** phi4-mini on one prompt (sample 2, seed 1002) falls into a repetition loop and Ollama aborts it ("token repeat limit reached"), every time, with or without logprobs. Re-seeding just that call would break the sampling protocol, so it is recorded and excluded: 1,199/1,200. (`run_config.json` → `summon.aborted`)
7. **Laptop memory pressure.** During the run the machine got low enough on RAM that the agent's background watcher process was killed by Claude Code. The generation run itself survived.
8. **ETA lies.** The runner's ETA started at 4.6 h because it averaged in the slowest model; the real time was ~2.5 h for the last 75%.
9. **Pending:** where TabPFN loses (Phase 5), LLM self-check result (Phase 5), any `global_freq` leakage measurement (Phase 5).

---

## 4. Honest edge cases worth a sentence

- **Import names that were never registered.** Two of the ghosts are the `import` names of real, popular tools whose PyPI distribution has a different name. The models wrote them in `pip install` lines and the names are unregistered, so anyone could claim them. They stay labelled as ghosts (that is exactly the attack), but they blur "invented" vs "confused". Don't name them in the post.
- **The top-15k rule will look strong.** No ghost is in the top-15k list, so "not in top 15k" has near-perfect recall. The real test for TabPFN is precision on the 11% of real names *outside* the list, and the unseen-names split.
- **`cross_model` and leave-one-model-out.** `cross_model` counts how many models suggested the name for the prompt; in split B that would include the held-out model. Phase 5 must recompute it from the training models only (or note the leak).
- **Scope cuts:** models 2–3 are general instruct models, not coder models (must fit 4 GB VRAM, three different vendors); 2 samples instead of 3; Ultramarine hosted arm waiting on an API key (provider waitlist). All in `run_config.json` → `cuts`.

---

## 5. Agent-session notes (for "My Agent Session")

- The agent (Claude Code) wrote the extractor test-first (43 tests before the code), built every stage and ran the study; 111 tests at the end of Phase 4.
- Human decisions: the ethics rules (no installs, no registrations, no public ghost lists), approving model swaps and scope cuts, the 25% pilot before the full run, and when to spend money.
- Real moments the agent caught: the thinking-only model, the three label bugs, the VRAM/context speedup, the deterministic aborted generation (diagnosed by replaying the exact call).

---

## 6. Still {{TODO}}

- Phase 5: arm × split metrics, CIs, calibration, τ, TabPFN timing, LLM self-check.
- Phase 6: how many invented names are registered today / match the squat rules.
- Phase 7: real hook demo (time, the blocked command) for the post's opening scene.
- Ultramarine arm: needs the key and spend approval.
- A real reaction from someone else: `{{TODO: real reaction}}`.
