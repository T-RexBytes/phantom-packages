# Improvements backlog (after all phases are done)

Written 2026-10-30, after Phase 5. **Not started.** When Phases 6–9 are finished, check each item
against the project's state at that time before doing it. Keep the v1 numbers (Phase 5 as committed in
`d8907bc`) and report any v2 next to them, saying what changed and why.

## Where things stand (v1, `results/summary.json`)

- TabPFN-3.5 AUROC 0.974 (split A), 0.978 (B), 0.948 (C, unseen names).
- Lead over gradient boosting is +0.006 (0.001–0.012) on A. On C it is not significant vs gradient
  boosting (+0.010, −0.001–0.022) or the edit-distance rule (+0.013, −0.001–0.028).
- Every arm hits the same precision ceiling: about 0.76 at 90% recall; recall at 95% precision ≤ 4%.
- TabPFN wins on calibration (ECE 0.017 vs 0.05–0.06) and loses badly on speed (about 20 s per test
  fold on CPU vs < 0.1 s for gradient boosting).

The ceiling is a feature problem, not a model problem: the false alarms are probably real niche
packages outside the top-15k list (77 real rows are outside it), which look like ghosts on every
name feature. Tuning TabPFN won't fix that.

## To do, in order

1. **Error analysis (≈15 min, terminal only).** At τ, list the false positives (real names flagged)
   and false negatives (missed ghosts) and group them by type: niche-but-real, affix pattern,
   truncated, import-name, and so on. This decides which features in item 2 are worth adding, and it is
   material for "What Went Wrong". Never commit the names.
2. **2–3 new generation-time features, fixed before looking at scores (≈1 h incl. rerun).** None may
   use registry data (keep `tests/test_leakage.py` green).
   - *Task-word overlap:* the share of the name's parts (split on `-`, `_`, `py` prefix) that appear in the
     prompt text. Hypothesis: ghosts are often task words glued together.
   - *Min token logprob* over the name's tokens (we only use the mean today; the logprobs are already stored).
   - *Generic affix on a non-package stem:* a `py` prefix or `-parser`/`-tools`/`-api`/`-sdk` suffix on a
     stem that isn't itself in the top-15k list.
   - Report as v2 next to v1, and say the features were chosen after the v1 error analysis.
3. **Guard speed (≈30 min, needed for Phase 7 anyway).** About 20 s per prediction on CPU is too slow for
   a hook. Measure:
   - the `v3.5-fast` weights;
   - `fit_mode="fit_with_cache"`;
   - one-row predictions.
   - A CUDA PyTorch build would use the RTX 3050, but it's a ≈2.5 GB download: **ask the user first.**
4. **Fair self-check baseline (≈10 min).** Score gemma4 by its probability of answering "no" (from the
   logprobs), not a hard yes/no. If it still loses, the result is stronger. Needs the logprobs kept
   with the cached answers; if they weren't stored, re-asking is about 5 min of local calls.
5. **More data, only if ahead of schedule (≈2 h of generation, overnight).** About 100 more L3 prompts →
   roughly 150–200 more ghost mentions (L3 ghost rate is 30%), narrowing the split-C intervals. A
   3rd sample per prompt adds less, because ghosts repeat within the same prompt.

## Don't do

- **Heavy TabPFN tuning or ensembling** (more estimators, TabPFN + gradient boosting average). Expected
  gain ≈0.003–0.005 AUROC, it needs nested CV to stay honest, and it doesn't make a better story.
- **Changing τ, splits or metrics after seeing results.**

## Point for the post

For the bare question "does this name exist?", a local snapshot of PyPI's full name list (a few MB) answers
exactly, with no ML. It can't be a feature because it is the label. TabPFN's honest role is:
- **calibrated risk while offline**, before the registry can be asked;
- **suspicion about names that exist but are new**, where existence proves nothing (the Phase 6 squat check).

Saying this openly makes the post more credible.
