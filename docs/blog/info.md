# Blog notes: work in progress

Running notes for the post (`docs/05_BLOG_POST_BLUEPRINT.md`). Facts only, each with its source.
Numbers here are working copies; the post must quote `results/summary.json` / `results/*.json`.
**No unregistered ghost names in this file** (it is public). Ghost names live in `private/` only.

Last updated: 2026-10-30 (pre-Phase-9 audit).

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

## 6. Phase 5 findings (evaluation, 2026-10-30)

Sources: `results/eval.csv`, `loo.csv`, `auroc_diff.csv`, `calibration.csv`, `rates.csv`, `model/threshold.json`, `summary.json`.

**Ghost rates with 95% CIs** (bootstrap by prompt): all 15.1% (12.0–18.5); gemma4 8.1% (5.0–11.3); phi4-mini 16.7% (12.5–21.0); qwen3 20.9% (16.1–26.3); L1 2.0% (0.3–4.2); L2 8.8% (4.9–13.6); L3 30.0% (23.7–36.5). gemma4's CI does not overlap qwen3's.

**AUROC, split A (by prompt, 888 rows, 173 ghosts):** TabPFN 0.974 (0.964–0.983), gradient boosting 0.969, logistic regression 0.967, edit-distance rule 0.966, top-15k rule 0.946, gemma self-check 0.849.

- **TabPFN wins, but narrowly.** Paired bootstrap, TabPFN minus gradient boosting: +0.006 (0.001–0.012) on A; +0.006 (0.000–0.012) on B.
- **On unseen names (split C) TabPFN's lead is not significant:** vs gradient boosting +0.010 (−0.001–0.022), vs the edit-distance rule +0.013 (−0.001–0.028).
- **A one-line rule is almost as good.** "Edit distance ≥ 2 from any top-15k name" (τ tuned on train) catches 94.8% of ghosts at 71.9% precision. TabPFN at the guard threshold: 90.2% recall, 75.7% precision.
- **Where TabPFN loses:** precision at 90% recall. On A it is 0.757 vs gradient boosting's 0.758; on C, 0.757 vs logistic regression's 0.763 and gradient boosting's 0.766; on B, 0.760 vs gradient boosting's 0.770. These are ties, not wins.
- **Speed: TabPFN loses badly.** About 20 s per test fold on CPU vs under 0.1 s for gradient boosting. Total TabPFN predict time was about 160 s for 8 folds, because the work happens at prediction time.
- **Where TabPFN clearly wins: calibration.** ECE 0.017 vs 0.05–0.06 for gradient boosting and logistic regression; Brier 0.049 vs 0.061–0.064. When it says 80–90% it is a ghost 88.7% of the time (62 rows). 639 of 888 rows get p < 0.1, and none of those are ghosts.
- **Nobody is precise.** Recall at 95% precision is ≤ 4% for every arm on A. The best pre-registry detector still raises ~1 false alarm per 3 flags at 90% recall. That's why the registry decides.
- **"Ask the model if it's lying" fails in the opposite way from the one you'd expect.** gemma4 said a ghost exists for only 2 of 162 ghost names. But it said 143 of 271 *real* packages (53%) don't exist. AUROC 0.849, precision at 90% recall 0.45. It isn't a detector; it's a "no" machine.
- **Ghosts never cross prompts.** 0 of 162 ghost names appear under more than one prompt (51 of 271 real names do). Every repeat is the same task summoning the same ghost again. So the unseen-names split has all 173 ghosts, and the real test is the 348 unseen real rows.
- **Leave-one-model-out is not harder than by-prompt** (TabPFN 0.978 pooled; held-out gemma4 0.985, phi4-mini 0.975, qwen3 0.972). One honest caveat: B shares prompts between train and test, so it tests a new *model*, not new tasks.
- **Guard threshold:** τ = 0.506 on out-of-fold TabPFN probabilities, giving recall 0.902 and precision 0.757 (206 flagged of 888).
- **TabPFN details:** v9.2.0, TabPFN-3.5 weights, CPU (torch CPU build), max 726 training rows, no subsampling, `ignore_pretraining_limits` not needed.

**More "What went wrong" material:**
- **The first TabPFN fit failed with a license error even though the key was set.** The project only loaded `.env` on demand, so TabPFN never saw `TABPFN_TOKEN`. It fell back to a browser login, which crashed on Windows (`WinError 10038`). Fix: load `.env` before importing TabPFN.

---

## 7. Phase 6 findings (squat check, re-checked 2026-10-29 18:57 UTC)

Sources: `results/squat_summary.json`, `results/squat_public.csv`.

- **No ghost got a body during the study.** Of 162 ghost names that returned 404 at labelling, 0 were registered by the re-check, so all 162 are still unregistered. The "{{N_REGISTERED}} of them now do" title (option A) doesn't apply; use title B (or another).
- **40 unregistered ghosts were summoned ≥2 times.** They are kept private; the public files carry only 8-character hashes (`results/anon_ghost_ids.txt`), and a CI test fails if any public file contains one of those names.
- **211 names re-checked:** 162 label-time ghosts, 42 real names TabPFN scored ≥ 0.5, 3 names first released after the suggesting model's release, and 4 that are both.
- **2 registered packages match all five squat rules** (≤3 releases, first release in the last 24 months, <500 downloads/month, suggested by ≥2 (model, sample) pairs).
  - Both have a normal description and project links: they look like legitimate small libraries. A rule match is not an accusation.
  - One was first published *after* the model that suggested it was released, so the model could not have learned it from that package.
  - TabPFN scored both as ghost-like (p ≈ 0.94–0.95) before knowing they exist. That is the "still flags it after a squatter gives it a body" angle, and it is also a false alarm on a probably-legit package. Say both.
  - Decide before publishing whether to name them in the post (the spec allows registered names). Safer: describe them without names.
- **7 names were "born after summoning"** (registered, but first released after the suggesting model's release date).
- **Disclosure:** a draft email to security@pypi.org is in `private/` and **not sent**. Only mention it in the post if it is actually sent.
- Nothing was installed, downloaded or executed; only metadata GETs, 211 to PyPI and 49 to pypistats.

---

## 8. Phase 7 findings (guard, hook, Action; 2026-10-30)

- **The guard uses its own TabPFN model.** It is TabPFN-3.5-fast trained on the 10 name features only, because a hook sees just a name, never how it was generated. Out-of-fold AUROC is 0.973 (0.963-0.982), not different from the Phase 5 context model (diff -0.001, CI -0.006 to +0.002). ECE is 0.011. τ = 0.506 gives recall 90.2% and precision 75.0%. (`summary.json` → `guard_model`)
- **Why not the spec's design:** the Phase 5 model fed default context values scored AUROC 0.971 and ECE 0.029, and took 17 s per prediction. Name-only 3.5-fast does as well and takes ~3.5 s. That is a good "What went wrong / what I changed" beat: the spec's design was slower *and* no better.
- **Hook latency** (cold process, CPU laptop), measured end to end with the JSON Claude Code sends:
  - a command with no install: ~1.1 s (mostly `uv run` startup; TabPFN is never imported);
  - installing real, established packages: ~1.8 s (one PyPI request per name, no model);
  - a ghost: ~12.7 s (blocked; ~5 s of that is importing torch, the rest is TabPFN computing P(ghost) for the message).
- **The model only runs when it can matter:** for ghosts, young packages, offline mode, or when PyPI can't be reached. Real packages never wait for it.
- **Offline mode works with the Wi-Fi off:** `phantom check --offline` gives the demo ghost P(ghost) = 0.997 (GHOST_LIKELY, exit 2) and gives `requests` and `holidays` 0.00.
- **Suggestion fix found in the demo:** the first suggestion for a `python-`-prefixed ghost was an unrelated name (edit distance 5). It now prefers the affix match (the real package without the prefix). That's a small "what went wrong".
- **Docs verified for the hook** (code.claude.com/docs/en/hooks, 2026-10-30): exit 2 blocks and stderr goes to Claude as the reason; on exit 0 Claude never sees stderr; `Bash|PowerShell` is an exact-name matcher; a timed-out PreToolUse hook does not block.
- **The Action runs without TabPFN** (base install). PyPI facts decide, and the reasons say "P(ghost) unavailable".
- **Demo ghost choice:** use a ghost that was suggested **once** (not a repeated one, rule 3). Keep its name out of the post *text*: the CI leak test hashes all 162 unregistered ghosts and will fail on it.

---

## 9. Phase 8 findings (figures + Ghost Registry site; 2026-10-30)

- **Figures** (`results/*.png`, dark Halloween palette validated for colour-blind separation): rates by model and by niche with 95% CIs, calibration, precision-recall curves, repeat histogram.
- **Repeat histogram facts** (`results/repeat_counts.csv`): 122 ghost names were suggested once, 32 twice, 6 three times, 1 four times, 1 five times. Every name suggested 3+ times came from two or more models; at 2x it was mostly one model.
- **The site** (`web/`) reads every number from `web/data.json`, which is built from `results/summary.json`. The graveyard has 162 anonymised stones (keyed HMAC id, domain, niche, count, models). There are no registered ghosts to name, because none got a body.
- **Honest box on the site:** where TabPFN wins (AUROC, calibration) and where it loses (precision at 90% recall, speed, not significant on unseen names).
- **Accessibility:** every text colour ≥ 4.6:1 on its surface, alt text built from the real numbers, a table view next to each chart, keyboard focus styles, `prefers-reduced-motion` respected, and no horizontal scroll at a true 360 px viewport (measured with an iframe; headless Chrome can't size a window that small).
- **What went wrong:** the first local preview server started in the repo root instead of `web/`. It was bound to localhost only, and I stopped it at once. A good reminder that `.env` and `private/` live next to the site.
- **Resolved: keyed ids.** Plain 8-character SHA-256 ids were confirmable: anyone who guesses a name, or reruns the same open models, could hash it and check the list. Public ids are now HMAC-SHA256 with a secret key, so a guess can't be checked. The two registered squat-rule matches are no longer named anywhere public. Good "what went wrong" beat: the spec's own anonymisation was the leak.
- **Site not hosted** (decision): `web/` stays in the repo for local viewing; screenshots can go in the post.

---

## 10. Pre-Phase-9 audit (2026-10-30)

- `uv run tasks.py all --fast` reran every non-generation stage from the cache in 345 s (spec target: < 10 min). Every result file came back identical except timings and timestamps: the pipeline is deterministic.
- Found and fixed: the gemma self-check contacted Ollama even when every answer was cached (`--fast` would fail with Ollama off), and the hook offered an unrelated "Did you mean" name for ghosts with no close real name (e.g. edit distance 11). It now suggests only affix matches or near-typos.
- Added `tests/test_results_consistency.py`: every headline number agrees across summary, CSVs, model files, site data and README (CI-safe).

---

## 11. Still {{TODO}}

- Phase 7 demo: record the GIF; note the real time and day for the post's opening scene.
- Ultramarine arm: needs the key and spend approval.
- A real reaction from someone else: `{{TODO: real reaction}}`.
