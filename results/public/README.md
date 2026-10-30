# Public evaluation dataset (no package names)

`dataset.csv` is the modelling table of the study (one row per model × prompt × suggested
name, 887 rows, 172 ghosts) with every package name replaced by a keyed id.
`uv run python experiments/12b_eval_public.py` reruns split A (by prompt) and split C
(unseen names) for TabPFN-3.5 and gradient boosting from this file alone.

| column | meaning |
|---|---|
| `name_id` | `HMAC-SHA256(PHANTOM_ANON_KEY, normalized name)[:8]`. Same name, same id; without the secret key a guessed name cannot be checked against it. |
| `model` | the local model that suggested the name (Ollama tag) |
| `prompt_id` | the coding task (`data/prompts.jsonl` ids, see `results/prompt_counts.csv`) |
| `fold` | split-A fold (GroupKFold by prompt, `results/folds.json`) |
| `y_ghost` | 1 if PyPI returned 404 at labelling time (the label; never a feature) |
| `n_gens` | generations of this (model, prompt) that mentioned the name; needed for `global_freq` |
| `len`, `n_parts`, `has_digit` | length, number of `-` parts, contains a digit |
| `in_top15k`, `nn_lev`, `nn_jw`, `nn_rank_log`, `affix_of_real` | relation to the top-15k PyPI names: member, edit distance and Jaro-Winkler to the nearest, log10 rank of the nearest, real name ± a common affix |
| `char_lp`, `char_lp_z` | character-trigram language model score (mean log-prob, z-score) |
| `consistency` | share of this model's samples for the prompt that suggested the name |
| `cross_model` | how many models suggested the name for the same prompt |
| `position` | first position of the name in the reply's install line |
| `source` | integer code: where the name was extracted from (`results/model/categories.json`) |
| `niche_level` | 1 everyday, 2 specialised, 3 obscure task |
| `domain` | integer code of the task domain (`results/model/categories.json`) |
| `name_logprob` | the model's mean token log-prob while writing the name |
| `model_code` | integer code of `model` (the `model` feature) |

`global_freq` is not stored: it must be computed per fold from training rows only
(`phantom.features.with_global_freq`, grouping by `name_id`), so test-fold names never inflate it.
Split C (unseen names) also needs `name_id`: a test row counts as unseen if its id never appears
in the training fold. No column comes from registry data (CLAUDE.md rule 7).
