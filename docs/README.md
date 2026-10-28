# Phantom Packages: HF26 Week 4 Kit

This kit is a complete plan to **research, build, write and submit** an entry for the DEV **Hacktoberfest 2026 Open-Source AI Challenge, Week 4: "Ghost in the Machine"**.

## ⏰ Deadline

| Source | Deadline |
|---|---|
| Challenge API `ends_at` | `2026-10-31T06:59Z`, which is **Fri Oct 30, 11:59 PM PDT** (Sat Oct 31, 12:29 PM IST) |
| Announcement post | "Oct 31, 11:59 PM PDT" |

**Plan for the earlier one.** Aim to publish by **Oct 30, 8:00 PM PDT** so you have a buffer.

- Challenge page: https://dev.to/challenges/hacktoberfest-week4-2026-10-26
- Required tag: `#hf26challenge`. Use `devchallenge` too.

## The idea in one breath

Local open-weight coding models tell you to `pip install` packages that **don't exist**. Those are the ghosts. Attackers register those names ("slopsquatting"), and the ghost gets a body.

**Phantom Packages** has three parts:

1. **Measure** how often local models summon ghosts.
2. **Train TabPFN** to recognise a ghost name **offline**, before any registry lookup.
3. **Ship a guard:** a CLI, a Claude Code hook and a GitHub Action that stop an AI agent from installing one.

> **"The model can dream a package. Only the registry can make it real."**

## Files in this kit (read in order)

| # | File | What it's for |
|---|---|---|
| 1 | [01_RESEARCH_BRIEF.md](01_RESEARCH_BRIEF.md) | All the research: rules, winners, Week 4 census, partner categories, sources |
| 2 | [02_IDEA_AND_STRATEGY.md](02_IDEA_AND_STRATEGY.md) | Why this idea, how it compares, prize strategy, win odds, risks |
| 3 | [03_BUILD_SPEC.md](03_BUILD_SPEC.md) | Full technical spec: architecture, schemas, algorithms, eval protocol, CLI, hook, Action, site, tests, acceptance criteria |
| 4 | [04_CLAUDE_CODE_PROMPTS.md](04_CLAUDE_CODE_PROMPTS.md) | Master prompt plus per-phase prompts and recovery prompts to paste into Claude Code |
| 5 | [05_BLOG_POST_BLUEPRINT.md](05_BLOG_POST_BLUEPRINT.md) | Section-by-section blog draft with placeholders, title bank, hooks, prize paragraphs |
| 6 | [06_EXECUTION_TIMELINE.md](06_EXECUTION_TIMELINE.md) | Hour-by-hour plan, cut lines, fallback plans |
| 7 | [07_DEMO_AND_ASSETS.md](07_DEMO_AND_ASSETS.md) | Demo GIF/video script, screenshot list, cover image, site copy |
| 8 | [08_SUBMISSION_CHECKLIST.md](08_SUBMISSION_CHECKLIST.md) | Pre-publish checklist and self-scoring against the judging rubric |
| — | [templates/CLAUDE.md](templates/CLAUDE.md) | Drop into the project root so Claude Code keeps the rules across sessions |
| — | [templates/seed_tasks.md](templates/seed_tasks.md) | 150 seed coding tasks across 15 domains, input for Phase 1 |
| — | [templates/settings.hook.json](templates/settings.hook.json) | Claude Code hook config snippet |
| — | [templates/phantom-guard.yml](templates/phantom-guard.yml) | GitHub Action template |

## Before you start (about 20 minutes)

- [ ] Install Ollama and pull the models: `ollama pull gemma4:e4b`, plus two coder models (see 03 §2).
- [ ] Install `uv` and Python 3.11+.
- [ ] TabPFN license: accept it at https://platform.priorlabs.ai/account/licenses and copy your token into `TABPFN_TOKEN`. Set `TABPFN_NO_BROWSER=1`.
- [ ] At https://hacktoberfest.com/my/promos, claim the **Ultramarine $5 credit** and the **free .tech domain**.
- [ ] Create an empty GitHub repo `phantom-packages` (MIT) and turn on GitHub Pages.
- [ ] Make sure DevRelay is logged in, so you can save and embed the agent session later. Judges like seeing it.
- [ ] Copy `templates/CLAUDE.md` into the new project folder, then paste the master prompt from file 04.
