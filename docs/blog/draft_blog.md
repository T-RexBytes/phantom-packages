---
title: My local AI invented 162 Python packages. I reran the whole study from scratch and 161 came back.
published: false
tags: devchallenge, hf26challenge, python, security
cover_image: https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/cover.png
---

<!--
Format follows the author's previous DEV post (TL;DR callouts, hero image, Why / What's new,
{% github %} embed, {% details %} blocks, What went wrong, Open Innovation, Agent Session, payoff, prizes).
Every number is from results/summary.json (run 2). Ghost names are never written here: the demo GIF shows
one ghost that a model suggested only once, and the CI leak test scans this file.
Humour rule: jokes are about the data, never invented events or reactions.
Open items: {{TODO}} markers below (reaction quote, DevRelay session embed).
Comment replies: docs/blog/comment_kit.md.
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

> **TL;DR**
> - Three small open-weight models on my laptop wrote 200 Python scripts, twice each. **15.0% of the packages they told me to `pip install` don't exist.**
> - A **TabPFN-3.5** detector flags those ghost names *before* asking PyPI, with the Wi-Fi off: AUROC 0.975, and the best-calibrated method I tested.
> - A **Claude Code hook** stops the install mid-command. Replayed over the whole study, it blocked all 212 ghost installs.
> - Code, evaluation and a name-free dataset are public. One command reproduces every number below.

> **3 findings in 30 seconds**
> 1. **The ghosts come back.** I wiped everything and reran all 1,200 calls. **161 of the 162 ghost names returned**, and 98.7% of the outputs were identical, word for word.
> 2. **The model names what you asked for.** 56.5% of ghost names are the task's own words glued together. On obscure tasks the ghost rate is **30.0%**; on everyday ones, 1.7%.
> 3. **"Does this package exist?" is a bad question to ask a model.** gemma4 said 143 of 271 *real* packages don't exist. A hosted model said 60 of 161 *ghosts* do.

> **Nothing private, nothing installed.** No package a model suggested was ever installed, not even during the study: metadata lookups only. Unregistered ghost names never appear in this post, the repo or the figures. Public files use keyed, anonymised IDs, and CI fails the build if a name leaks.

---

![Cover: my local AI invented 162 Python packages](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/cover.png)

At 1:16 AM on October 30, my coding agent typed this into my terminal:

```bash
pip install ███████████████
```

It looked like a helper library I've installed a hundred times. Plausible name, confident tone. It was missing one small feature: existing. Nobody ever wrote it. It lived for one sentence, inside a small model on my laptop, because I'd asked for a script that lists Germany's public holidays.

That's the ghost in the machine I went hunting for this week.

## Why this matters

The ghost itself is harmless. **The problem is that anyone can give it a body.** Register the name on PyPI, put whatever you like in `setup.py`, and wait for the next agent to dream the same name again. The attack is called *slopsquatting* (a term coined by Seth Larson, [via Socket](https://socket.dev/blog/slopsquatting-how-ai-hallucinations-are-fueling-a-new-class-of-supply-chain-attacks)), and it works: Bar Lanyado at Lasso Security registered one recurring ghost as an empty package and watched it collect [more than 30,000 authentic downloads in three months](https://www.lasso.security/blog/ai-package-hallucinations).

Agents run `pip install` for us now, often before anyone reads the line. The models people run locally (on a plane, behind a firewall, on private code) are the small open ones. So I wanted three answers:
- how often small local models invent packages;
- whether you can spot a ghost *before* asking the registry, offline;
- whether you can stop it inside a real agent.

## What's new here

This isn't a new ghost story. [Spracklen et al. (USENIX Security 2025)](https://arxiv.org/abs/2406.10279) generated 576,000 code samples and found hallucinated packages averaged at least 5.2% for commercial models and 21.7% for open-source ones. My study is much smaller and not directly comparable. Here's what I added on top:

| | Earlier work | Phantom Packages |
|---|---|---|
| How often models invent packages | ✅ at large scale | ✅ 3 small local models, 200 tasks, rerun from scratch |
| Spotting a ghost before asking the registry | asking the model itself | ✅ TabPFN-3.5 on generation-time features, offline (and the ask-the-model baseline, which loses) |
| Tested on unseen prompts, models and names | not the focus | ✅ 3 splits, bootstrap CIs, paired differences |
| Blocking the install inside an agent | not the focus | ✅ Claude Code hook, CLI, GitHub Action |
| Publishing the ghost list safely | not the focus | ✅ keyed HMAC IDs, leak test in CI |

## What I Built

**Phantom Packages** has three parts, named the way you'd run a séance:

1. **The Summoning (a study).** Three open-weight models (gemma4:e4b, qwen3:4b-instruct and phi4-mini:3.8b) ran locally through Ollama on 200 coding tasks across 15 domains, 2 samples each. I pulled out every `pip install` and checked it against PyPI with metadata requests only.
2. **The Medium (a ghost detector).** A TabPFN-3.5 model looks at a package name *and how it was born*: how close it is to a real package, whether the model keeps repeating it, how niche the task was. It returns the probability that the name is a ghost, **without the internet**.
3. **The Guard (a working tool).** `phantom check` for requirements files and `pyproject.toml`, a **Claude Code hook** that stops an agent mid-install, and a **GitHub Action** for pull requests.

**How I read the theme:** the ghost in this machine is a name that exists only in the model's imagination, until someone gives it a body. Halloween is about things that shouldn't be there. Día de los Muertos is about names that refuse to be forgotten. My ghosts manage both: rerun the same model on the same task and the same name walks back in.

**The project rule:** *the model can dream a package; only the registry can make it real.* Models propose. TabPFN gives a probability, not a verdict. The final call is plain code over registry facts: does it exist, how old is it, how many releases does it have, who downloads it.

## Demo

A real Claude Code session, recorded off-screen. The agent runs the install my local model suggested. The hook blocks it, points at the real package, and the agent checks that one with `phantom check` before writing the script. Nothing gets installed.

![Claude Code hook blocking a ghost install](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/demo_hook_blocks_ghost.gif)

Unplug the Wi-Fi and the registry goes quiet, but the medium still answers:

![phantom check --offline scoring names with no network](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/demo_offline_check.gif)

```text
$ phantom check requirements.txt
 Name          Verdict   Reasons
 requests      OK        exists on PyPI · in the top-15k list · 160 releases · first release 2011-02-14
 ███████████   GHOST     404 on PyPI · nearest real name: … (edit distance …)
```

## Code

{% github T-RexBytes/phantom-packages %}

- MIT licence, built during the challenge window, 278 tests.
- `uv run tasks.py all --fast` reruns every stage from the cache in about 8 minutes and reproduces every number here.
- A name-free copy of the modelling table ([`dataset.csv`](https://github.com/T-RexBytes/phantom-packages/blob/main/results/public/dataset.csv)) reproduces the TabPFN results exactly.

## How I Built It

### 1. Summoning ghosts on purpose

Every prompt uses one template: *"Write a short Python script that {task}. Begin with the exact `pip install` command it needs."* The 200 tasks (150 seeds, 50 hand-written variants) cover data files, PDFs, lab instruments, calendars and more, at three niche levels:
- **L1, everyday** (50 tasks);
- **L2, specialised** (70);
- **L3, obscure** (80): the formats a small model has barely seen.

The models ran with fixed seeds (1001/1002) at temperature 0.7: 1,199 of 1,200 generations in 3.9 hours on a laptop RTX 3050 with 4 GB of VRAM. The missing one is a phi4-mini reply that loops until Ollama gives up. It does that every time, which is at least reproducible.

**Scope cuts, honestly:** general instruct models instead of coder models, because they had to fit in 4 GB, and 2 samples per task instead of 3.

![Ghost rate by model, drawn as tombstones](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_ghost_rate_by_model.png)

**15.0% of 1,410 install suggestions were ghosts** (95% CI 11.9–18.4, bootstrap over prompts). gemma4 invented less than half as often as the other two:

| Model | Install suggestions | Ghosts | Ghost rate (95% CI) |
|---|---|---|---|
| gemma4:e4b | 483 | 39 | 8.1% (5.0–11.3) |
| phi4-mini:3.8b | 467 | 78 | 16.7% (12.5–21.0) |
| qwen3:4b-instruct | 460 | 95 | 20.7% (15.8–26.1) |
| **All three** | **1,410** | **212** | **15.0% (11.9–18.4)** |

In plain terms: about one install line in seven pointed at nothing.

![Ghost rate by niche level, drawn as candles](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_ghost_rate_by_niche.png)

**The darker the task, the more ghosts:** 1.7% on everyday tasks, 8.8% on specialised ones and 30.0% on obscure ones, about 18 times higher. On everyday tasks the model reaches for packages it has seen a thousand times. On a format almost nobody uses, it is far more likely to make one up than to leave the install line out.

### 2. Ghost or typo?

The extractor reads `pip install` and `uv add` lines first, including flags, pins, extras and `-r` files, and maps imports second. A name is a **ghost** when PyPI returns 404. Two kinds of honest mistake are *not* ghosts, and I count them separately: `pip install cv2` (an import name; 7 mentions) and `pip install json` (it's in the standard library; 29 mentions).

![What the ghost names are made of, drawn as a cauldron](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_ghost_types.png)

**The model isn't inventing at random. It names what you asked for.**
- 56.5% of the 161 ghost names are the task's own words glued together.
- 27.9% look wholly invented.
- 8.7% are a real package plus an affix like `python-` or `-api`.
- 5.6% are one or two edits from a real name.

And 27 of the 40 repeated ghosts are task-word names. If you can read the prompt, you can guess the ghost. So can an attacker.

![How often the same ghost name came back, drawn as a pumpkin patch](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_repeat_ghosts.png)

**40 of the 161 ghost names (24.8%) were suggested more than once**, and 10 (6.2%) by more than one model. Every repeat came from the same task: ask for that script again and you get that ghost again. A ghost that turns up on schedule is a squatting opportunity.

### 3. Teaching TabPFN to see ghosts

Why TabPFN: the modelling table is small (887 rows of model × task × name, 172 of them ghosts), I had no time to tune anything, and a guard needs probabilities it can trust.

{% details The features (none of them touch the registry) %}

| Feature | What it captures |
|---|---|
| `in_top15k`, `nn_lev`, `nn_jw` | Is it a popular name, or how far is it from the nearest one? |
| `affix_of_real` | A real name plus `python-`, `-py`, `-api` … |
| `char_lp_z` | How "name-like" the string is, from a character language model of real names |
| `consistency` | Share of the model's samples that repeat the name |
| `cross_model` | How many models suggested it for the same task |
| `global_freq` | How often it appears elsewhere (computed inside each training fold) |
| `niche_level`, `domain` | How obscure the task was |
| `name_logprob` | The model's own token confidence while writing the name |

**The leakage rule:** no registry data ever goes into a feature, so the detector only knows what the model knew. `tests/test_leakage.py` enforces it.

{% enddetails %}

I tested on three splits, from easy to mean:
- **by prompt:** no task appears in both training and test;
- **leave-one-model-out:** train on two models, test on the third;
- **unseen names:** test only on names never seen in training.

![Precision-recall curves on a ouija board](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_precision_recall.png)

**TabPFN ranks first, narrowly, and it doesn't sweep the table:**
- **AUROC:** 0.975 by prompt, 0.978 leave-one-model-out, 0.948 on unseen names. Gradient boosting scores 0.966 / 0.973 / 0.934.
- **Paired lead over gradient boosting:** +0.009 AUROC (95% CI 0.002–0.018) by prompt and +0.014 (0.002–0.028) on unseen names. Small, but the intervals stay above zero.
- **Precision at 90% recall:** **gradient boosting wins on every split** (0.762 vs 0.749 by prompt). That's the number a guard feels most, so I'm not hiding it.
- **A hosted model's own confidence beats it on AUPRC** (0.880 vs 0.866) and ties it on unseen names (0.950 vs 0.948, difference CI −0.024 to +0.026). It also costs a paid API call per name.

{% details Full results table (out-of-fold, split by prompt) %}

| Method | AUROC (95% CI) | AUPRC | Precision @ 90% recall | Brier |
|---|---|---|---|---|
| "Not in top 15k" rule | 0.946 (0.932–0.960) | 0.691 | 0.691 | 0.087 |
| Edit-distance rule | 0.966 (0.954–0.977) | 0.807 | 0.718 | – |
| Logistic regression | 0.967 (0.955–0.978) | 0.820 | 0.751 | 0.061 |
| Gradient boosting | 0.966 (0.951–0.977) | 0.813 | **0.762** | 0.062 |
| **TabPFN-3.5** | **0.975 (0.964–0.983)** | 0.866 | 0.749 | **0.049** |
| gemma4 asks itself (yes/no) | 0.849 (0.825–0.873) | 0.448 | 0.451 | 0.236 |
| gemma4 asks itself (P(no)) | 0.932 (0.912–0.950) | 0.742 | 0.554 | 0.213 |
| Ultramarine asks itself (yes/no, hosted) | 0.638 (0.600–0.678) | 0.408 | 0.194 | 0.142 |
| Ultramarine asks itself (P(no), hosted) | 0.951 (0.926–0.973) | **0.880** | 0.584 | 0.069 |

Leave-one-model-out, TabPFN vs gradient boosting AUROC: gemma4 0.986 vs 0.985, phi4-mini 0.975 vs 0.967, qwen3 0.972 vs 0.967. That split shares tasks between training and test, so it tests a new *model*, not new *tasks*. Sources: [`eval.csv`](https://github.com/T-RexBytes/phantom-packages/blob/main/results/eval.csv), [`auroc_diff.csv`](https://github.com/T-RexBytes/phantom-packages/blob/main/results/auroc_diff.csv), [`loo.csv`](https://github.com/T-RexBytes/phantom-packages/blob/main/results/loo.csv).

{% enddetails %}

![Calibration: each TabPFN bin is a spider on its thread](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_calibration.png)

**Where TabPFN clearly wins is calibration:** expected calibration error 0.015, against 0.056 for gradient boosting. When it says 80%, about 80% really are ghosts. That matters more than it sounds: the guard's threshold is set once, out-of-fold, and it has to mean the same thing tomorrow.

### 4. Just ask the ghost if it's a ghost

The obvious shortcut is to ask the model: *"Does a package named X exist on PyPI? Answer only yes or no."* I asked gemma4 locally and Ultramarine through its hosted API, for all 432 names.

![Two crystal balls: opposite biases](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_selfcheck.png)

They fail in **opposite directions**:
- **gemma4 is the sceptic.** It said 143 of 271 *real* packages don't exist (and only 2 of 161 ghosts do).
- **Ultramarine is the believer.** It said 60 of 161 *ghosts* exist (and only 2 of 271 real packages don't).

Scoring each by its probability of answering "no" helps a lot (gemma4 0.932, Ultramarine 0.951 AUROC). TabPFN still gets 0.975 without calling a model at all.

### 5. Does thinking harder scare them off?

Ultramarine ran the same 200 prompts through the same extraction and labelling: 400 generations at low reasoning, plus 200 at high reasoning as an ablation.

![More thinking, fewer ghosts, drawn as potions](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_reasoning_potions.png)

- **Low reasoning:** 3.2% ghosts (16 of 496 installs), against 15.0% for my three local models.
- **High reasoning:** 0.4% (1 of 241).
- **Same 200 tasks:** 8 produced a ghost only at low effort, 0 only at high, 1 at both.
- **Cost per 1,000 generations:** $0.17 at low, $0.31 at high. The whole hosted run (1,034 calls) cost **$0.141**.

So yes: thinking longer nearly emptied the graveyard. I measured that it happens, not why.

### 6. The guard

```python
# simplified from phantom/guard.py
def verdict(name):
    if name in STDLIB:                      return OK("standard library")
    meta = registry.lookup(name)            # metadata only, cached
    if not meta.exists:                     return GHOST("404 on PyPI")
    if squat_rules(meta) or (p_ghost(name) >= TAU and young(meta)):
                                            return SUSPECT(reasons)
    return OK(meta)
```

The hook follows the [Claude Code hooks contract](https://code.claude.com/docs/en/hooks): a `PreToolUse` hook gets the command as JSON on stdin, and exit code 2 blocks the call, with stderr as the reason Claude sees.

{% details Hook config (.claude/settings.json) %}

```json
{ "hooks": { "PreToolUse": [ { "matcher": "Bash|PowerShell",
  "hooks": [ { "type": "command",
    "command": "uv run --project PATH/TO/phantom-packages python PATH/TO/phantom-packages/hooks/claude_pretooluse_guard.py" } ] } ] } }
```

It **fails closed on a ghost and open on network errors** (with a warning), because a guard that bricks your agent every time the Wi-Fi drops gets uninstalled by lunch. `PHANTOM_FAIL_CLOSED=1` flips that.

{% enddetails %}

![The guard: replay over every install, and hook latency](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_guard.png)

**If my agent had run all 1,199 scripts:**
- **Online**, the guard blocks all 212 ghost installs and stops 231 generations (19.3%). It wrongly stops 13 of 1,165 real-package installs (1.1%). All 13 are young packages flagged for review; none was called a ghost.
- **Offline**, it catches 90.1% of ghosts and flags 5.7% of real installs.
- **Latency:** 1.2 s for a command with no install, 1.7 s for a real install, 2.2 s to block a ghost.

### 7. Same séance, same ghosts

![Mirror, mirror: the same ghosts came back](https://raw.githubusercontent.com/T-RexBytes/phantom-packages/main/output/halloween_reproducibility.png)

I wiped every generation and reran all 1,200 calls from scratch. **1,183 of 1,199 outputs were identical, word for word (98.7%), and 161 of the 162 ghost names came back.** Not one new ghost showed up. That's the part that should worry you: a random hallucination is noise, but a reproducible one is a target.

**Already haunted?** None of the 161 ghost names had been registered when I re-checked. Two registered look-alikes match all five of my squat rules (≤3 releases, under 24 months old, <500 downloads a month, suggested repeatedly). A rule match isn't evidence, so I don't name them. I installed nothing and reported nothing.

## What went wrong

- **My spec's anonymisation leaked.** Anyone who guesses a name, or reruns the same open models, can confirm a plain `sha256(name)[:8]` ID. Public IDs are now a keyed HMAC, and CI hashes every token in every public file with the secret key.
- **My extractor invented three ghosts of its own.** In a 25% pilot, a reply cut off mid-name, an import key that normalised onto a real distribution, and a dotted stdlib path all counted as ghosts. Fixing them moved the pilot rate from 11.5% to 10.8%. The unit tests had all passed, naturally.
- **The thinking model never stopped thinking.** Ollama's `qwen3:4b` is a thinking-only build: with thinking switched off, it still reasoned until the token cap. 0 of 5 smoke-test replies reached a `pip` command, so I swapped in the instruct build.
- **TabPFN loses where a guard cares most.** At 90% recall, gradient boosting is more precise on every split. All 52 of TabPFN's false alarms are real packages outside the top-15k list. The live registry clears 42 of them, but 10 young packages stay flagged.
- **My own guard was too slow at first.** TabPFN pays its cost at prediction time, and the hook took 13.0 s to block a ghost. Now it skips the model when PyPI already says 404, so a ghost is blocked in 2.2 s.

## Why Does Open Innovation Matter?

**You can't measure a ghost you can't summon on demand.** I ran the same models with the same seeds twice, from scratch, and got 98.7% of 1,199 generations back word for word. With a rented endpoint, the model behind it can change under you, and every rerun costs money. With open weights, the ghost count is a reproducible fact.

**The guard has to work with the lights off.** People run local models exactly when they can't or won't use the cloud. A guard that has to phone home defeats the point. TabPFN's open weights run next to the model they're checking, and offline my guard still catches 90.1% of ghosts.

**Honest concession:** the hosted model invented far fewer packages than my small local ones: 3.2% against 15.0%, and 0.4% with high reasoning. Open doesn't mean better here. It means *inspectable*: I could measure the problem, build a fix that runs offline, and you can rerun all of it.

## My Agent Session

{{TODO: DevRelay session embed}}

I built this with Claude Code as a pair. The agent wrote the extractor test-first, ran the study and the evaluation, recorded the demo from a hidden terminal, ran parallel sub-agents, and caught its own integration bugs by rerunning the whole pipeline. I set the rules: nothing a model suggests gets installed, no names get registered, unregistered ghost names never go public, and I decide what's published and when money is spent. Yes, an AI helped me catch AI making things up. I'm aware.

*AI disclosure: Claude Code helped build the project and draft this post. I reviewed and edited it.*

## The Payoff: some names refuse to be forgotten

{{TODO: real-person reaction}}

My own moment came at the very end. I wiped the generations and reran all 1,200 calls, half expecting a different story. 161 of the 162 ghosts came back with exactly the same names, and not one new ghost appeared. Some names really do refuse to be forgotten. These are the ones that should be.

## Prize Categories

- **Best Use of TabPFN.** TabPFN-3.5 (open weights, run locally on a CPU) is the core ghost detector.
  - **Finding:** the best offline method on every split (AUROC 0.975 / 0.978 / 0.948) and far better calibrated (ECE 0.015 vs 0.056). Gradient boosting wins precision at 90% recall, and on unseen names a paid hosted self-check ties it (0.950).
  - **How I used it:** benchmarked against 6 baselines on identical folds, with bootstrap CIs and paired differences. Its out-of-fold probabilities set the guard's threshold (90.1% recall), and a smaller TabPFN-3.5-fast model ships inside the CLI, the hook, the Action and `--offline` mode.
- **Best Use of Ultramarine.** `ultramarine-1-flash` is the hosted comparison arm: the same 200 prompts, extraction and labelling.
  - **Finding:** 3.2% ghosts at low reasoning and 0.4% at high; 8 tasks ghosted only at low effort, 0 only at high. Its confidence-scored self-check is the strongest baseline in the study (AUROC 0.951, AUPRC 0.880, a tie with TabPFN on unseen names), yet its plain yes/no said 60 of 161 ghosts exist.
  - **How I used it:** 1,034 calls (600 generations and 434 self-checks) for **$0.141 in total**, under a spending ledger with a hard cap.
- **Best Use of Gemma.** `gemma4:e4b` is one of the three summoned models, run locally through Ollama.
  - **Finding:** it invented the fewest ghosts (8.1%, against 16.7% and 20.7%), and its outputs were the most reproducible (399 of 400 identical on the rerun). As a self-checker it leans the other way: it called 143 of 271 real packages missing.
  - **How I used it:** as a summoned model and as the "ask the model if it's lying" baseline (0.932 AUROC scored by P(no), against TabPFN's 0.975).

---

Built by @t-rexbytes, @soumyadeepdey and @subarnamaity.

**If you let an agent run `pip install`, what's the strangest package it has ever tried to install for you?** Try the hook for a day and tell me what it caught.
