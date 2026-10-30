# Lessons from the build: what the plan didn't know

Things I had to discover, tweak or do differently from the spec while building Phantom Packages.
Each item: what happened → what I changed → why it matters. Facts only, with numbers where they exist.
No unregistered ghost names appear here (this folder is public and leak-tested).

---

## Running local models (Ollama, 4 GB laptop GPU)

1. **A "thinking" model that never answers.**
   - **What happened:** Ollama's `qwen3:4b` is the thinking-only 2507 build. With `think: false` it still reasoned in plain text until it hit the token cap, and 0 of 5 smoke-test replies contained a `pip` command.
   - **Change:** swapped it for `qwen3:4b-instruct`. The lesson is to smoke-test every model on the real prompt before a long run.
2. **VRAM, not compute, was the bottleneck.**
   - **What happened:** with Ollama's default context, phi4-mini ran 36% on the CPU.
   - **Change:** set `num_ctx` to 1,024. Prompts are about 90 tokens and replies at most 350, so this keeps every model on the GPU. qwen3 went from 21 to 30 tok/s and phi4-mini from 25 to 35 tok/s.
3. **A talkative model.** gemma4:e4b hit the 350-token cap in 80% of replies. The extractor had to learn to drop the unfinished last line of a cut-off reply; otherwise half a package name became a fake ghost.
4. **A failure you can't retry away.** phi4-mini on one prompt (p112, sample 2, seed 1002) loops until Ollama aborts it ("token repeat limit reached"). It did so with and without logprobs, and again in run 2. Re-seeding just that call would bias the protocol, so it's recorded as aborted: 1,199 of 1,200.
5. **Long runs need to survive things.** The first full run died at generation 73 when Ollama went offline. I added "stop after 3 consecutive failures" and resume-by-key, then tested the resume by killing a run after 4 generations: 0 duplicates. Later runs were launched detached, so closing the editor session or a memory-pressure clean-up couldn't kill them.
6. **The ETA lies early.** The runner averages per-model latency, so it said 4.6–6 h when the truth was about 3.6–3.9 h. The slowest model goes first and skews it.
7. **Same seeds really do give the same text.** A from-scratch rerun (run 2) produced 1,183 of 1,199 generations byte-for-byte identical (98.7%). 161 of 162 ghost names came back; the one that didn't had appeared only once.

## Labelling ghosts

8. **Three labelling bugs found in real data, not in the 43 unit tests,** on the 25% pilot:
   - a truncated last line;
   - an import key that normalised onto a real distribution;
   - a dotted stdlib path inside a `pip install` line.

   Fixing them moved ghost mentions from 41 to 39, the pilot ghost rate from 11.5% to 10.8%, and module-name confusions from 8 to 2. Running a pilot before the full grid paid for itself.
9. **Import names that were never registered.** Some ghosts are the `import` name of a real tool whose PyPI distribution has another name. The models put them in `pip install` lines and the names are unregistered, so they're claimable. They stay labelled as ghosts, because that is exactly the attack, but they blur "invented" vs "confused".
10. **Ghosts never cross prompts.** No ghost name appeared under two different prompts, while real names often do. That made the "unseen names" split contain every ghost, so the real test there is the unseen real names.

## TabPFN on a CPU laptop

11. **The licence check needs `.env` loaded first.** The first fit failed with a licence error even though `TABPFN_TOKEN` was set: the project only loaded `.env` on demand. TabPFN then fell back to a browser login, which crashed on Windows (`WinError 10038`). Fix: load `.env` before importing TabPFN.
12. **TabPFN's cost is at prediction time.** Fitting takes 0.3 s; predicting 200 rows takes about 20 s on the CPU, because it attends over the whole training set every time. That doesn't matter for a 5-fold evaluation (2.5 minutes), but it's fatal for a hook.
13. **The spec's guard design was slower and no better.** Feeding the full model "default" generation context gave AUROC 0.971 and ECE 0.029 at about 17 s per prediction. A name-only TabPFN-3.5-fast model scored 0.973 and ECE 0.011 at about 3.5 s. The guard never sees how a name was generated, so name-only is also the more honest design.
14. **Then the hook skipped the model altogether when the registry already knows.** PyPI's 404 decides on its own, so TabPFN now runs only offline, when PyPI is unreachable, or for young packages. Measured with `experiments/14_hook_latency.py`:
    - a ghost install is blocked in 2.2 s instead of 13.0 s;
    - a real install is allowed in 1.7 s;
    - a command with no install passes in 1.2 s.
15. **A fair baseline changes the story's size, not its direction.** Scoring gemma4's "does it exist?" by its probability of saying "no" (top-20 first-token logprobs) instead of the yes/no word lifted it from AUROC 0.849 to 0.932. It still loses to TabPFN on every split. The surprising part is that the model says "no" to 143 of 271 real packages.

## Privacy and ethics

16. **The spec's anonymisation leaked.** Plain `sha256(name)[:8]` IDs can be confirmed: anyone who guesses a name, or reruns the same open models, can hash it and check the list. The public IDs are now a keyed HMAC with a secret key (local `.env`, a GitHub secret in CI). The CI leak test hashes every token in public files with that key, and skips rather than passes when the key is absent.
17. **Not naming legitimate packages.** Two registered packages match every squat rule but look legitimate. A rule match is not evidence of anything, so they're described, never named, in public files.
18. **A rebuild must never spend money.** The evaluation stage would have quietly asked the hosted model about new names because an API key was present. Paid calls now need an explicit `--hosted-spend` flag.
19. **Partial results are not results.** The paused hosted run (79 of 400 generations) almost reached the charts and site. Hosted numbers are now gated on completeness.
20. **Eval etiquette.** The hosted provider asks to be emailed before evals. The run is strictly sequential with backoff, and they were told first.

## Hosted model (Ultramarine)

21. **The logprob stream starts with a `</think>` token.** Reading P(no) at position 0 gave nonsense, scoring a hugely popular package as probably fake. P(no) is now read at the first token after `</think>`.
22. **Low reasoning effort is cheap.** About 275 output tokens per generation (1,099 of 21,736 tokens were reasoning), about $0.00015 per call. The whole hosted plan's worst case is under $1 at the published $0.15/$0.50 per million tokens.

## Building the demo and the site

23. **You can't screen-record a GPU terminal by window title.** Windows Terminal captured as a black frame. Capturing the screen region instead recorded whatever window was on top, including the user's browser. Those frames were deleted.
    - **Fix:** run the real Claude Code session in a hidden pseudo-terminal (ConPTY via pywinpty), feed it to a VT emulator (pyte), and render every screen change to video with real timing. The desktop isn't involved.
24. **Small things the recorder had to learn:**
    - the recorder's own `VIRTUAL_ENV`/`PATH` leaked into the recorded shell;
    - Claude Code's verbose mode is needed to show the hook's message;
    - a font's "missing glyph" box isn't reported as missing;
    - Claude Code's prompt suggestion uses a non-breaking space;
    - a dimmed suggestion renders at full brightness without dim support, so it looked typed.
25. **Cut the boring parts honestly.** Claude Code's spinner and timer change every second, so "nothing happening" is not "nothing changed". Idle time is cut by ignoring status-line-only changes and holding each real screen for its reading time. The hook demo went from 43 s to about 22 s, and the MP4s stay real-time.
26. **Specificity bugs hide in screenshots.** A `.wrap { padding: … }` shorthand silently overrode section spacing and the hero's side gutters. Logical properties (`padding-inline` / `padding-block`) fixed it.
    - Headless Chrome can't size a window below about 500 px, so a 360 px "phone" check has to run inside an iframe.
27. **Never serve a site from the repo root.** The first local preview server started in the repo root, so `.env` and `private/` were reachable on localhost for a minute. It now always runs with `--directory web`.
28. **Validate chart layout automatically.** The Halloween figures' first pass had 13 overlapping or off-canvas labels. A validator now checks every label's bounding box and the "keep-out" decorations (moon, tree, house) on every render.

## Working with parallel agents

29. **Shared files need one owner.** Agents worked in parallel on separate files. The orchestrator owned `tasks.py`, the README and the integration. Two integration bugs still slipped through and were caught by rerunning the whole pipeline from cache:
    - a changed function signature;
    - a file one stage expected but no stage wrote.

    `tasks.py all --fast` reruns 15 stages in about 8 minutes and reproduces every result, and it became the integration test.
30. **A consistency test is worth more than another unit test.** `tests/test_results_consistency.py` checks that every headline number agrees across the summary, the CSVs, the model files, the site data and the README. It caught the README still quoting run 1's numbers after run 2.

## Late lessons (pre-publish)

31. **A validator only knows what it measures.** Matplotlib counts an annotation's arrow as part of its text box,
    and some display fonts (Eater) report a box twice the height of the glyphs. Both raised false "overlaps".
    The fix was to draw arrows as separate, unlabelled annotations and to measure each font before spacing it.
32. **Bold is a claim too.** The results table bolded TabPFN's AUPRC out of habit; a hosted baseline was higher.
    Check every highlighted cell against the numbers, not just the numbers themselves.
33. **"Best on every split" needs every arm in the comparison.** Adding the hosted self-check after the post was
    written quietly broke one sentence (0.950 vs 0.948 on unseen names). Re-read claims when you add an arm.
34. **Same tasks or it isn't an ablation.** The headline low-reasoning rate (3.2%) pools two samples; the
    high-reasoning arm ran one. The fair comparison uses sample 1 on the same 200 tasks (3.6% vs 0.4%), and the
    figure says so.
35. **Ship the fonts.** Halloween figures that depend on Windows-only fonts don't reproduce elsewhere. The repo now
    carries OFL/Apache fonts with their licences. Three figures still use Windows display fonts (Chiller,
    Viner Hand, Jokerman); elsewhere `fp()` falls back to a repo font.

