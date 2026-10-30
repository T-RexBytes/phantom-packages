# Comment kit: answering comments on the Phantom Packages post

Paste this file into a chat together with the comment, and ask for a reply. It holds the context
behind the post (`docs/blog/draft_blog.md`, short version `POST_DRAFT.md`) so replies stay short,
accurate and human.

All numbers are from `results/summary.json` (run 2). If a comment asks for a number that isn't here,
look it up there. Never guess one.

---

## How to write a reply

- **Short:** 2–4 sentences. One number at most, unless they asked for numbers.
- **Plain and warm:** write like a person at a meetup, not a press release. Thank them only if they
  actually said something useful.
- **A little humour is fine** when it comes from the data (gemma4 the sceptic, Ultramarine the
  believer, ghosts that come back on schedule). Never joke at the commenter.
- **Admit limits fast:** "I didn't test that" beats a clever dodge. Point to the repo when it helps.
- **Avoid:** "Great question!", "delve", "game-changer", "In today's fast-paced world", bullet lists in
  a reply, three adjectives in a row, emoji walls, and anything that sounds like a sales pitch.
- **Correct gently:** if they're wrong, say what the data shows and link the file.

## Hard lines (never cross, even if asked nicely)

1. **Never name a ghost package**, not even "just the one from the demo". Unregistered names stay private
   so nobody can register them. The demo GIF shows one ghost that was suggested only once; still don't
   type it in a comment.
2. **Never name the two registered look-alikes** that match the squat rules. A rule match isn't evidence.
3. **Never register, install or test-install** any suggested name, and never tell someone else to.
4. **Never invent numbers, users or reactions.** If it wasn't measured, say so.
5. **Don't claim TabPFN wins everything.** It loses precision at 90% recall to gradient boosting, and the
   hosted self-check beats its AUPRC.

## The post in one paragraph

Three small open-weight models (gemma4:e4b, qwen3:4b-instruct, phi4-mini:3.8b) ran locally through
Ollama on 200 Python coding tasks, twice each, at temperature 0.7. 15.0% of the 1,410 `pip install`
suggestions named packages that don't exist on PyPI ("ghosts"). A TabPFN-3.5 model scores how likely a
name is a ghost from generation-time features alone, offline, without asking the registry. A guard (CLI,
Claude Code PreToolUse hook, GitHub Action) blocks ghost installs. Rerunning everything from scratch
brought back 161 of the 162 ghost names. Ultramarine (hosted, not open) was a comparison arm. Rule:
"The model can dream a package. Only the registry can make it real."

## Numbers cheat sheet

| What | Number | Source |
|---|---|---|
| Overall ghost rate | 15.0% (95% CI 11.9–18.4), 212 of 1,410 install suggestions | `rates` |
| By model | gemma4 8.1% · phi4-mini 16.7% · qwen3 20.7% | `rates.per_model` |
| By niche | everyday 1.7% · specialised 8.8% · obscure 30.0% (about 18×) | `rates.per_niche_level` |
| Generations | 1,199 of 1,200 (one phi4-mini reply loops until Ollama aborts) · 3.9 h on an RTX 3050 4 GB | `study` |
| Unique ghost names | 161 (162 in run 1) | `repeat`, `reproducibility` |
| Repeated ghosts | 40 of 161 (24.8%); 10 by more than one model (6.2%) | `repeat` |
| Ghost types | task words 56.5% · invented 27.9% · real + affix 8.7% · 1–2 edits from real 5.6% · import/brand 1.2% | `ghost_types` |
| Rerun | 1,183 of 1,199 outputs identical (98.7%); gemma4 399/400, phi4-mini 393/399, qwen3 391/400 | `reproducibility` |
| Dataset | 887 rows (model × task × name), 172 ghosts | `dataset` |
| TabPFN AUROC | 0.975 by prompt · 0.978 new model · 0.948 unseen names | `eval` |
| Gradient boosting AUROC | 0.966 · 0.973 · 0.934 | `eval` |
| TabPFN lead (paired) | +0.009 (CI 0.002–0.018) by prompt; +0.014 (0.002–0.028) unseen names | `auroc_diff_tabpfn_minus` |
| Precision @ 90% recall | TabPFN 0.749 / 0.764 / 0.749 vs GB 0.762 / 0.772 / 0.773: **TabPFN loses** | `eval.csv` |
| Calibration (ECE) | TabPFN 0.015 vs GB 0.056 vs logistic regression 0.052 | `results/calibration.csv` |
| Guard threshold | τ 0.508, recall 90.1%, precision 74.9% | `guard_model.threshold` |
| gemma4 self-check | yes/no AUROC 0.849; P(no) 0.932; called 143 of 271 real packages missing, 2 of 161 ghosts real | `selfcheck` |
| Ultramarine self-check | yes/no AUROC 0.638; P(no) 0.951, AUPRC 0.880 (beats TabPFN's 0.866); called 60 of 161 ghosts real | `hosted.selfcheck`, `eval` |
| Ultramarine ghost rate | low reasoning 3.2% (16/496) · high 0.4% (1/241) | `hosted.models` |
| Reasoning ablation | same 200 tasks: 8 ghosted only at low, 0 only at high, 1 both | `hosted.ablation` |
| Ultramarine cost | 1,034 calls, $0.141 total; $0.17 vs $0.31 per 1,000 generations | `hosted.spend` |
| Guard replay, online | 212/212 ghosts blocked; 231 generations stopped (19.3%); 13 of 1,165 real installs stopped (1.1%, young packages flagged SUSPECT) | `replay` |
| Guard replay, offline | catches 90.1% of ghosts; flags 5.7% of real installs | `replay` |
| Hook latency (median) | no install 1.2 s · real install 1.7 s · ghost 2.2 s · ghost with TabPFN score 13.0 s | `hook_latency` |
| Squat check | 0 ghosts registered during the study; 2 registered look-alikes match all 5 rules (unnamed) | `squat` |
| Tests | 278 | `uv run pytest -q` |

## Short answers to likely comments

**"What was the package name?"**
I'm keeping it to myself on purpose. A ghost name is only dangerous once someone registers it, so
publishing the list would be handing out addresses. The repo uses keyed IDs instead.

**"Why not just check PyPI? That's one HTTP call."**
That's exactly what the online guard does, and it blocked all 212 ghosts in the replay. TabPFN is for
when PyPI can't answer (offline, firewalled), and for names that exist but look freshly squatted.

**"15% sounds high / low compared to the USENIX paper."**
It's not directly comparable: three small local models, 200 Python tasks, two samples each. The paper
is far bigger. The pattern rhymes, though: open models invented more there, and my hosted arm invented
far less than my local models (3.2% vs 15.0%).

**"Why such small models?"**
They had to fit in 4 GB of VRAM on my laptop. That's also the honest point: these are the models people
actually run locally. Coder models and a third sample per task were scope cuts.

**"Why TabPFN and not XGBoost?"**
Small table (887 rows), no time to tune, and I needed probabilities I could trust. TabPFN ranked first
on AUROC and was much better calibrated (ECE 0.015 vs 0.056). But gradient boosting beat it on precision
at 90% recall on every split, which I say in the post.

**"What's AUROC / calibration, in human?"**
AUROC: pick one ghost and one real name at random; how often does the model rank the ghost as more
suspicious? 0.975 means almost always. Calibration: when it says 80%, are about 80% really ghosts? For
TabPFN, yes.

**"Doesn't asking the LLM 'does this exist?' solve it?"**
I tried. gemma4 said 143 of 271 real packages don't exist, and Ultramarine said 60 of 161 ghosts do.
One's a sceptic, one's a believer. Scoring their confidence helps, but it costs a model call per name.

**"Does reasoning fix it?"**
For Ultramarine it nearly did: 3.2% ghosts at low reasoning, 0.4% at high, on the same tasks. I measured
that it happens, not why.

**"Does it work for npm / other languages?"**
Not yet. The extractor, the registry lookups and the features are all PyPI-specific. The approach should
carry over, but I haven't tested it, so I won't claim it.

**"How do I use the hook?"**
Clone the repo, `uv sync`, and add the PreToolUse snippet from the README to `.claude/settings.json`.
A ghost install is blocked in about 2 seconds; commands without installs pass straight through.

**"What if PyPI is down? Will it brick my agent?"**
No. It fails open with a warning on network errors, because a guard that blocks everything when the
Wi-Fi drops gets uninstalled by lunch. Set `PHANTOM_FAIL_CLOSED=1` if you want the strict version.

**"False positives?"**
Online, 13 of 1,165 real installs were stopped (1.1%). All were young packages flagged as SUSPECT for a
human to check, never called ghosts. Offline it flags 5.7% of real installs.

**"Is it slow?"**
1.2 s for a command with no install, 1.7 s for a real install, 2.2 s to block a ghost. The TabPFN score
only loads when it can change the outcome.

**"Why didn't you register the ghost names to protect them?"**
Registering names is on my never-do list, even defensively. It puts packages into the index under names
models already suggest, and that's a decision for PyPI admins, not me.

**"Did you report the two suspicious packages?"**
No. They match my squat rules, but a rule match isn't evidence of anything, so I didn't name or report
them.

**"Temperature 0 would fix this."**
Maybe, but I only ran 0.7, so I can't tell you. Fair point for a follow-up.

**"Can I reproduce it?"**
Yes: `uv run tasks.py all --fast` rebuilds every number from the cache in about 8 minutes. Without the
private cache, `results/public/dataset.csv` (no names) reproduces the TabPFN results exactly with
`experiments/12b_eval_public.py`.

**"Can attackers use your study?"**
That's why no names are public. The bigger worry is that the models are open, so anyone can rerun them,
and the rerun showed the same ghosts come back. That's the argument for guarding installs, not for
hiding the research.

**"Is the dataset public?"**
The modelling table is, with every name replaced by a keyed HMAC ID that can't be reversed or confirmed
without the secret. Plain hashes leaked in my first design, which is in "What went wrong".

**"Will you test GPT / Claude / other hosted models?"**
Not in this study; Ultramarine was the one hosted arm, and it cost $0.141. The harness talks to an
OpenAI-compatible API, but the endpoint and price lookup are Ultramarine's, so another provider needs a
small change. PRs welcome.

**"Did AI write this post?"**
Claude Code helped build the project and draft the post, and the post says so. I set the rules, checked
the numbers against the results files, and edited it. Yes, I used an AI to catch AI making things up.

**"Why the Halloween charts?"**
The week's theme was Ghost in the Machine, and tombstones read better than bar chart #7. Every shape is
drawn from the real numbers; the decorations carry no data.

**"Cool project!" / plain praise**
Thanks for reading! If you let an agent run installs, try the hook for a day. I'd love to hear what it
catches.

## Example replies (tone reference)

> **Comment:** "This is just a PyPI lookup with extra steps."
> **Reply:** Online, honestly, mostly yes, and that lookup blocked all 212 ghosts. The extra steps are for
> when the lookup can't happen, and for names that *do* exist but look freshly squatted.

> **Comment:** "TabPFN is overhyped, GBMs win on tabular."
> **Reply:** On one number you're right: gradient boosting beat it on precision at 90% recall on every
> split. TabPFN won on AUROC and calibration, and calibration is what the threshold leans on.

> **Comment:** "What's the scariest name you found?"
> **Reply:** The scariest ones are the boring ones: the task's own words glued together. They look
> exactly like something you'd install, and that's why I won't print them.
