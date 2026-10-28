# 02: Idea and Strategy

## 1. How the shortlist compared

Each idea was scored 1–5 against what the research says wins: little Week 4 competition, a literal fit with the theme, clean labels for TabPFN, novelty on DEV, risk of a 2.5-day build, and "wow".

| Idea | Week 4 competition | Theme fit | Data / labels | Novelty on DEV | Build risk | Wow | **Total /30** |
|---|---|---|---|---|---|---|---|
| **Phantom Packages** (AI-invented pip packages + TabPFN + guard) | 5 (0 entries) | 5 ("the ghost is the AI's imagination") | 5 (registry returns 404 or 200) | 3 (topic is known on DEV, but not this angle) | 3 | 5 | **26** |
| Ghost Job Exorcist (TabPFN + Stagehand) | 5 | 3 (a pun) | 2 (no public ghost-job labels; proxy only) | 3 | 3 | 4 | 20 |
| Dependency death forecast | 2 (pip-seance, Revenant, Phantom Dependency) | 5 | 5 | 2 | 3 | 3 | 20 |
| Zombie APIs: AI code calling deprecated functions | 5 | 4 | 4 (run the code and see) | 4 | 1 (needs a sandbox) | 4 | 22 |
| Whisper ghost voices (EVP) | 2 (Sổ Giỗ, spirit box) | 5 | 3 | 2 | 3 | 4 | 19 |

**Decision: Phantom Packages.**

## 2. The idea, fully

### Problem
Coding agents increasingly run `pip install` **on their own**. Language models sometimes name packages that don't exist. Attackers watch for repeated hallucinated names and register them with malicious code. This is **slopsquatting**: the ghost gets a body. Small local open-weight models are what people run in offline or private agents, and they are the least studied.

### What we build
1. **The Summoning (study).** Run 3 local open-weight models on about 200 realistic Python tasks, 2–3 samples each.
   - Extract every package the model asks you to install.
   - Check each one against PyPI: a **404 means a ghost**.
   - Report ghost rate per model and per task niche, how often ghosts repeat, and which ghosts **already have bodies** (registered by someone).
2. **The Medium (TabPFN).** Train TabPFN-3.5 to predict *P(ghost)* from **generation-time features only**: name shape, distance to real names, how often the name comes back across samples, model, and how niche the task is.
   - Benchmark it against rules, logistic regression, gradient boosting, and Gemma asking itself "does this exist?".
   - Run **leave-one-model-out** to see if it catches a model it never saw.
3. **The Guard (product).** `phantom check` gives GHOST / SUSPECT / OK.
   - It also ships as a **Claude Code PreToolUse hook** that blocks the install, and a **GitHub Action** for pull requests.
   - **`--offline` mode** uses TabPFN alone, so it works on a plane or behind an air gap. That's the "lights (and the internet) off" line from the challenge prompt.
4. **The Ghost Registry.** A static Halloween site with the numbers and a "graveyard" of tombstone cards, on a `.tech` domain.

### The one rule (use it as an H3 aphorism in the post)
> **"The model can dream a package. Only the registry can make it real."**

- The LLM proposes names.
- TabPFN gives a calibrated risk.
- Deterministic code and registry facts decide.
- Nothing is ever installed.

### Why it fits "Ghost in the Machine"
- **Spooky, literally:** the ghost is a name that lives only inside the machine's imagination, until something evil gives it a body.
- **Honors the past (optional):** a section on how package names, like the dead, persist. Repeated hallucinations are echoes of old training data (old tutorials, renamed packages). Show 2–3 ghosts that turn out to be **real packages that were renamed or removed years ago**. The model remembers a dead package. That's a powerful moment if your data shows it. Check the PyPI history and old docs.

## 3. Positioning against prior DEV posts

Prior posts (listed in 01 §5) mostly explain slopsquatting, scan repos, or check whether a name exists.

**Our differentiators. Say them in one sentence near the top of the post:**
> "Most slopsquatting tools ask the registry. Phantom Packages asks *before* the registry: a TabPFN model that spots a ghost name from how it was born, offline, and still flags it after a squatter gives it a body."

Then credit the prior posts by name in *How I Built It* or *What I Measured*. Judges reward honesty.

## 4. Prize strategy

You can **win only once**, so enter every category you genuinely use. Each partner paragraph must be concrete.

| Category | Prize | Competition | Our use | Enter? |
|---|---|---|---|---|
| **Overall** | $250 | ~390 entries | Writing + theme + measured study + product | ✅ (automatic) |
| **TabPFN** (featured) | $200 | ~5–8 (Titanic Séance is strong) | Core detector, baseline table, leave-one-model-out, calibration, offline guard | ✅ **primary** |
| **Ultramarine** | $100 | ~0 | Hosted comparison arm: same prompts, ghost rate plus **reported cost**. Optionally writes the per-ghost "case files" from the evidence JSON. | ✅ if you claim the $5 credit |
| **Gemma** (featured) | $200 | 80–110 | One of the 3 summoned models, plus the self-check baseline | ✅ (low odds, no cost) |
| **.tech domain** | $100 | 3–5 | The Ghost Registry live at `<name>.tech` | ✅ if you claim it (about 15 min of work) |
| **GitHub Copilot** | $100 | 3–6 | Only if you actually build with Copilot. The Action alone is a weak claim. | ⚠️ optional |
| Sentry Agent Tracing | $100 | 8–12 | Could trace the summoning runs. Skip unless you have time. | ❌ by default |

## 5. Win probability (honest estimate)

| Target | Executed to the winners' standard* | Weak execution |
|---|---|---|
| Overall | 10–15% (Sổ Giỗ is very strong) | <3% |
| TabPFN featured | 35–50% | ~10% |
| Ultramarine | 50–70% (if the use is meaningful and cost is reported) | ~25% |
| .tech | 20–35% | ~10% |
| **At least one prize** | **≈55–70%** | ≈15–25% |

\*"Winners' standard" means:
- an eval table with baselines
- a held-out split
- a calibration plot
- a "What went wrong" section with ≥3 real failures
- a live demo (site plus GIF)
- every template heading kept
- an opening scene
- an aphorism rule
- an honest "where closed is better" concession
- about 2,000–3,000 words

These are judgment calls based on the competitor counts and past winners, **not guarantees**.

## 6. Risks and what to do about them

| Risk | Impact | Mitigation |
|---|---|---|
| Generation is slow on your laptop | Fewer rows for TabPFN | Resumable runs. Cut to 120 prompts × 2 samples × 3 models (about 720 generations). Use `num_predict` ≈ 350. |
| Too few ghosts (small class) | Weak classifier | Weight the tasks toward niche ones (niche_level 3). Add samples at temperature 1.0 as a separate arm. Optionally add names from the depscope open dataset (check its CC-BY-NC-SA license) **as a separate evaluation only**. |
| TabPFN doesn't beat gradient boosting | Weaker TabPFN story | Report it honestly. Lean on calibration, small-data behaviour and leave-one-model-out. Winners admitted losses (19/20). |
| Label leakage (`in_top15k` alone is a strong signal) | Inflated metrics | Report it as its own baseline arm. Also show results **excluding** names that are in the top 15k. |
| Import names differ from distribution names | Wrong labels | Use `pip install` lines first, apply the import→distribution map, and keep unmapped imports out of the main labels. |
| Publishing an attacker shopping list | Ethical / reputational | Keep unregistered repeated names in `private/`. Public files show aggregates and already-registered names only. Disclose to security@pypi.org if something looks malicious. |
| TabPFN license / token friction | Blocked | Accept the license first (before you start). Fall back to `tabpfn-client`. |
| Overlap with prior DEV slopsquatting posts | Lower creativity score | Cite them; lead with the offline TabPFN and leave-one-model-out angle. |
| Deadline confusion | Missed submission | Publish by **Oct 30, 8 PM PDT**. |

## 7. The real-person moment (the "Spirit of It" section)

Winners show a real reaction, and DEV praises it every week. Pick one:
- **A friend who uses AI coding agents** installs the hook, then watches it block a ghost during a real session. Screenshot their message.
- **Your own agent session**: the hook stops Claude Code mid-task. Screenshot the blocked command and the agent's reaction ("I'll use `X` instead").
- **A maintainer or security person** you know reads the numbers. Quote their one-line reaction.

Leave `{{TODO}}` in the draft until it's real. **Never invent a reaction.**
