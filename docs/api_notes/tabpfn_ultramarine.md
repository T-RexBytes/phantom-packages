# API notes: TabPFN and Ultramarine (checked 2026-10-28)

I checked everything here against live sources on 2026-10-28. Nothing was installed.
**UNVERIFIED** means I could not confirm the point from an official source.

---

## 1. TabPFN (local package)

### 1.1 Package, version, Python

| Item | Value | Source |
|---|---|---|
| PyPI name | `tabpfn` | https://pypi.org/pypi/tabpfn/json |
| Latest version | **9.2.0**, uploaded 2026-10-12T15:12:54Z. Earlier: 9.1.0 (2026-10-02), 9.0.0 (2026-09-15). None yanked. | https://pypi.org/pypi/tabpfn/json |
| `requires_python` | `>=3.10` | same |
| Python classifiers | 3.10, 3.11, **3.12, 3.13**, 3.14. Both 3.12 and 3.13 are supported. | same |
| Default model | **TabPFN-3.5** (one multitask checkpoint for classification and regression). The `TABPFN_MODEL_VERSION` setting defaults to `ModelVersion.V3_5`. | https://github.com/PriorLabs/TabPFN , https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/settings.py |
| Code license | Apache-2.0 | https://github.com/PriorLabs/TabPFN |
| Weights license | TabPFN-2.5/2.6/3/3.5 weights are **non-commercial**. Our research use is fine. | https://github.com/PriorLabs/TabPFN , https://docs.priorlabs.ai/models |

Key dependencies (`requires_dist` for 9.2.0): `torch>=2.5`, `safetensors>=0.4.0`, `numpy>=1.21.6`,
`scikit-learn>=1.2.0`, `scipy>=1.11.1`, `pandas>=1.4.0`, `einops>=0.4.0`, `huggingface-hub>=0.23.0`,
`pydantic>=2.8.0`, `pydantic-settings>=2.10.1`, `joblib>=1.2.0`, `tqdm>=4.66.0`, `filelock>=3.11.0`,
`lightgbm>=4.4`, `skrub>=0.7,<0.11`, `matplotlib>=3.6.1`, and optionally `wandb` (extra `wandb`).
There is no upper bound on torch. The FAQ says "TabPFN-3 requires PyTorch 2.5 or above; we recommend 2.8 or above" (https://docs.priorlabs.ai/faq).

On Windows with NVIDIA, the README says to install CUDA PyTorch first and then `tabpfn`. For CPU-only, install CPU PyTorch first to save disk space (https://github.com/PriorLabs/TabPFN).
With uv, that means pointing `torch` at the PyTorch CUDA or CPU index in `pyproject.toml`.
- **UNVERIFIED:** I did not check the exact uv index config for torch on Windows.

### 1.2 Auth, license, weights

How it works (source: https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/browser_auth.py and the README):

1. Downloading gated weights (v2.5, v2.6, v3, v3.5, v3.5-fast) calls `ensure_license_accepted(hf_repo_id)`.
2. The token is resolved in this order:
   - the `TABPFN_TOKEN` env var (this wins);
   - the cached `~/.cache/tabpfn/auth_token`;
   - the fallback `~/.tabpfn/token`, which tabpfn-client uses.
3. The token is checked against `https://api.priorlabs.ai/protected/` (Bearer).
4. License acceptance is checked at `https://api.priorlabs.ai/account/license/?version=<license_name>`, using the JSON field `accepted`.
5. With no usable token, a browser login opens at `https://ux.priorlabs.ai/login` (or `/accept-license`) with a localhost callback.
6. `TABPFN_NO_BROWSER` set to any value except `0/false/no/off` turns off the browser flow. A missing token then raises an error telling you to set `TABPFN_TOKEN`.

The base URLs can be changed with `TABPFN_AUTH_GUI_URL` (default `https://ux.priorlabs.ai`) and `TABPFN_AUTH_API_URL` (default `https://api.priorlabs.ai`) (settings.py).

**Headless setup for this project:**
1. Log in at https://ux.priorlabs.ai, open the **License** tab, and accept the TabPFN-3.5 license.
2. Copy your token into `.env`: `TABPFN_TOKEN=...` and `TABPFN_NO_BROWSER=1`.
   - TabPFN settings also read `.env` (README, settings.py).
   - The Prior Labs JWT goes only to `api.priorlabs.ai`. Datasets stay local and are never sent (README).

License text for TabPFN-3.5: https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE

**Weights download** (`model_loading.py`, README):
- Weights are fetched on the first `fit()` with `huggingface_hub.hf_hub_download`. A `FileLock` prevents parallel downloads.
- Files:
  - `tabpfn-v3.5-20260909.safetensors` serves both classifier and regressor.
  - `tabpfn-v3.5-fast-20260909.safetensors` is the Fast variant.
- Cache directory on Windows: `%APPDATA%\tabpfn\`. If `APPDATA` is missing, it falls back to `./.tabpfn_models` with a warning.
- To change it, use **`TABPFN_MODEL_CACHE_DIR`** or `model_path=` in the constructor.
- After the weights are downloaded, TabPFN works offline.
- **UNVERIFIED:** whether you also need a Hugging Face login or `HF_TOKEN` for the gated repo. The code sends `HF_TOKEN` to huggingface.co if one is present, but the README says accepting the Prior Labs license is enough.

Other useful env vars (README / settings.py):
- `TABPFN_ALLOW_CPU_LARGE_DATASET`: lifts the CPU sample cap.
- `TABPFN_MAX_BATCHED_TEST_ROWS`: default 32768.
- `PYTORCH_CUDA_ALLOC_CONF`: default `max_split_size_mb:512`.
- Windows crash workaround: for `Windows fatal exception: code 0xc000001d`, set `ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16`.
- The settings source has no telemetry setting.

### 1.3 Classifier API

Import: `from tabpfn import TabPFNClassifier`. Constructor (keyword-only, from
https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/classifier.py):

| Arg | Default | Notes |
|---|---|---|
| `n_estimators` | `"auto"` | Uses the checkpoint's own count, raised on wide data. An explicit int is never overridden. |
| `categorical_features_indices` | `None` | 0-based hints. `None` lets the model infer. Pandas `category` columns and string columns are treated as categorical. Don't one-hot encode. |
| `device` | `"auto"` | Tries CUDA, then MPS, then CPU. Also accepts `"cpu"`, `"cuda"`, or a list for multi-GPU. |
| `ignore_pretraining_limits` | `False` | Skips the size and CPU-cap errors. The class limit cannot be skipped. |
| `random_state` | **`0`** | A fixed seed, unlike sklearn's `None`. Results are not guaranteed identical across hardware. |
| `model_path` | `"auto"` | Downloads the default model (3.5). Can be a local path or `ModelSpecs`. |
| `softmax_temperature` | `"auto"` | Uses the checkpoint value. Only affects `predict_proba`/`predict_logits`. |
| `balance_probabilities` | `False` | Rescales by class frequency in post-processing. |
| `average_before_softmax` | `False` | |
| `inference_precision` | `"auto"` | Pass a torch dtype (e.g. `torch.float32`) for reproducibility. |
| `fit_mode` | `"fit_preprocessors"` | `"low_memory"`, `"fit_with_cache"` (fast repeat predicts, high memory), or `"batched"`. |
| `memory_saving_mode` | `"auto"` | |
| `n_preprocessing_jobs` | `1` | `n_jobs` is deprecated. |
| others | | `auto_scale_n_estimators`, `enable_torch_compile`, `keep_cache_on_device`, `kv_cache_precision`, `inference_config`, `eval_metric`, `tuning_config`, `show_progress_bar`, `differentiable_input` |

Methods: `fit(X, y) -> Self`, `predict(X)`, `predict_proba(X) -> np.ndarray`.

Choosing a model version. The `ModelVersion` values are `V2="v2"`, `V2_5="v2.5"`, `V2_6="v2.6"`, `V3="v3"`, `V3_5="v3.5"`, `V3_5_FAST="v3.5-fast"`
(https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/constants.py):

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

clf = TabPFNClassifier(device="auto", random_state=0)            # default = TabPFN-3.5
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="cpu")  # explicit
clf_fast = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)
clf.fit(X_train, y_train)            # first call downloads weights
proba = clf.predict_proba(X_test)    # shape (n, n_classes)
```
The signature is `create_default_for_version(cls, version: ModelVersion, **overrides)`.
Passing constructor kwargs through `**overrides` follows from that signature, but I have not seen it in an example. **UNVERIFIED**.

**Size limits** (https://docs.priorlabs.ai/models, README):

| Model | Max rows | Max features | Classes |
|---|---|---|---|
| TabPFN-3.5 / 3.5-Fast | 1,000,000 | 20,000 (6,000 recommended) | more than 160 needs the many-class extension |
| TabPFN-3 | 1,000,000 | 2,000 | 160 |
| TabPFN-2.5 / 2.6 | 100,000 | 2,000 | 10 |
| TabPFN-2 | 10,000 | 500 | 10 |

**CPU cap:** `cpu_sample_limit()` returns **5000** training samples for v3, v3.5 and v3.5-fast, and 1000 for v2.x.
Going above the cap raises an error unless you set `ignore_pretraining_limits=True` or `TABPFN_ALLOW_CPU_LARGE_DATASET=1`
(https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/inference_config.py).

### 1.4 Save and load a fitted model

```python
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model

save_fitted_tabpfn_model(clf, "models/tabpfn_guard.tabpfn_fit")   # suffix .tabpfn_fit required
clf2 = load_fitted_tabpfn_model("models/tabpfn_guard.tabpfn_fit", device="cpu")
```

Signatures: `save_fitted_tabpfn_model(estimator, path) -> None` and `load_fitted_tabpfn_model(path, *, device="auto")`
(https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/model_loading.py).

**Format.** The file is a zip with three parts:
- `init_params.json`
- `fitted_attrs.joblib`
- `executor_state.joblib`

It **omits the foundation weights**, so the loading machine needs the cached weights or a token to download them. The joblib parts are pickle-based, so load only trusted files.

**Moving between CPU and GPU works.** Tensors are moved to CPU before saving, and on load the model calls `est.to(device)`. A model fitted on GPU can be loaded with `device="cpu"`.

Weights only: `save_tabpfn_model(model, path)` saves a torch checkpoint, and `load_model_criterion_config` reads it back.
Raw `pickle` of an estimator is not a documented path. Use the functions above.

### 1.5 Performance and hardware (RTX 3050 4 GB laptop)

- From the FAQ (https://docs.priorlabs.ai/faq):
  - "At minimum, TabPFN requires an NVIDIA T4 GPU to run efficiently". A100/H100 are recommended.
  - "CPU ... only suitable for small datasets".
- The README says older GPUs with about 8 GB VRAM work well, and some large datasets need 16 GB.
- The OOM page (https://docs.priorlabs.ai/troubleshooting/OOM-errors) gives VRAM for TabPFN-3.5 on an A100 with default estimators, 200 features and 1,000 test rows:

| Training rows | VRAM |
|---|---|
| 1k–10k | about 3 GB |
| 50k | about 3.5 GB |
| 100k | about 4 GB |
| 200k | about 7 GB |

- `fit_mode="fit_with_cache"` costs about 4 GB **per estimator** at 1M rows. Fast costs about 2 GB per estimator. Neither figure fits a 4 GB card at scale.
- Ways to cut memory: `memory_saving_mode=True`, predict in chunks of about 1000 rows, and `inference_config={"SUBSAMPLE_SAMPLES": ...}`.
- **RTX 3050 4 GB: UNVERIFIED.** The docs don't mention 4 GB cards.
  - From the ~3 GB figure, a dataset of ≤10k rows with a few dozen features *should* fit. This is a guess.
  - Plan: smoke-test `device="cuda"`, fall back to `device="cpu"`, and keep the training set ≤5000 rows on CPU (or set `ignore_pretraining_limits=True`).
- Do one predict call per batch. Each `predict` re-processes the training set (README).

### 1.6 tabpfn-client (hosted fallback)

| Item | Value | Source |
|---|---|---|
| PyPI name / version | `tabpfn-client` **0.7.0**, uploaded **2026-10-28T15:20:23Z** (today). `requires_python>=3.10`. | https://pypi.org/pypi/tabpfn-client/json |
| Import | `from tabpfn_client import init, TabPFNClassifier, TabPFNRegressor` | https://github.com/PriorLabs/tabpfn-client |
| Auth | Env var `TABPFN_TOKEN`, or `tabpfn_client.set_access_token("...")`. `interactive_login()` is opt-in. `init()` never prompts and raises `RuntimeError` if no token is set. Keys come from https://platform.priorlabs.ai/account/api-keys. | same |
| Model | `model_path="auto"` gives v3 (README). v3.5 is opt-in through `model_path="v3.5_default"` or `model_version="v3.5"`, "on accounts with access" (changelog 0.5.2). `v3.5-fast` was added in 0.6.0. | https://raw.githubusercontent.com/PriorLabs/tabpfn-client/main/CHANGELOG.md |
| Persistence | `save_model()` / `load_model()` (0.6.x). These are server-side fits: `FittedModelNotFoundError` is raised if the server has dropped the fit. | changelog |
| Limits / cost | Limits are set by the server (`/tabpfn/get_settings`) and reset daily at 00:00 UTC. Use `estimate_cost()` to estimate cost. No numbers are published. Uploads and standard fits are not charged. Predictions use up token budget (FAQ). | README, https://docs.priorlabs.ai/faq |
| Privacy | "your data will be sent to our servers". This is fine for our features, which contain no secrets, but note it. | README |

Differences from the local package:
- The client has no `device` argument.
- Local text/datetime handling is described for the local package.
- Full parity of constructor arguments is **not documented** (UNVERIFIED).
- Whether one token works for both local and client is **UNVERIFIED**. Both read `TABPFN_TOKEN`, and local auth falls back to the client's `~/.tabpfn/token`, which suggests they are the same account token.

---

## 2. Ultramarine (Paper Instruments)

The official docs exist and match the spec. Index: https://docs.paperinstruments.com

| Item | Value | Source |
|---|---|---|
| Base URL | `https://api.paperinstruments.com/v1` (matches the spec) | https://docs.paperinstruments.com/docs/ultramarine |
| Model id | `ultramarine-1-flash` (matches the spec) | same |
| Auth | `Authorization: Bearer $PAPERINSTRUMENTS_API_KEY`. Keys are `pi_live_...` (production) or `pi_test_...` (non-production). Errors: `401 invalid_api_key` / `401 missing_api_key`. | https://docs.paperinstruments.com/docs/ultramarine/authentication |
| Compatibility | Works with the OpenAI SDK (chat completions, responses) and the Anthropic SDK (messages) | index |
| `reasoning_effort` | **Supported**: `low`, `high`, `max`. **Default `max`**. Any other value, or turning thinking off, returns `400`. | https://docs.paperinstruments.com/docs/ultramarine/chat-completions |
| Token limit param | `max_completion_tokens` (includes reasoning). `max_tokens` is an alias; send only one. Default output ceiling is 65,536. | same |
| Sampling defaults | `temperature` 1 (0–2), `top_p` 0.95, `top_k` -1. `seed`, `n` (1–16), `stop` (≤16), `response_format` (json_object/json_schema/text), `logprobs`/`top_logprobs` (0–20) | same |
| Rejected | `developer` role, audio, legacy `functions`, `store`, `metadata`, `service_tier`. Unknown fields return `400` with `error.param`. | same |
| Response | `choices[0].message.content` and `message.reasoning_content`. `finish_reason` can be `"length"`. Header `x-request-id`. | same |
| Usage fields | `prompt_tokens` (includes cached), `prompt_tokens_details.cached_tokens`, `completion_tokens` (includes reasoning), `completion_tokens_details.reasoning_tokens`, `total_tokens`. **No cost field.** In streams, usage needs `stream_options={"include_usage": true}`. | same |
| Context window | **UNVERIFIED** in the official docs. Third-party posts say 131,072. | https://wavect.io/de/blog/ultramarine-1-flash-api-document-workflows/ (third party) |
| Rate limits | Set per key. Headers `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`. On 429, honour `retry-after` and `x-should-retry: false`. `GET /v1/usage` returns your limits. | https://docs.paperinstruments.com/docs/ultramarine/rate-limits , https://docs.paperinstruments.com/docs/ultramarine/errors |

### 2.1 `GET /v1/pricing`: exists, no auth needed

The docs say it returns current rates without authentication (https://docs.paperinstruments.com/docs/ultramarine/billing).
I fetched `https://api.paperinstruments.com/v1/pricing` on 2026-10-28. Response, verbatim:

```json
{"object":"price_card","model":"ultramarine-1-flash","version":"metronome-6451b310-6801-48d7-8dad-fc9746a5f9f7","effective_at":1792555882,"unit":"per_1m_tokens","uncached_input_usd":"0.15","cached_input_usd":"0.03","output_usd":"0.50","uncached_input_nanodollars_per_token":150,"cached_input_nanodollars_per_token":30,"output_nanodollars_per_token":500,"web_search_per_1k_usd":"10.00","web_search_nanodollars_per_call":10000000}
```

- The USD prices are **strings**. The `*_nanodollars_per_token` fields are integers (1 USD = 1e9 nanodollars). Compute cost from the integers.
- Reasoning tokens are billed at the output rate.
- Cost per call in nanodollars is:
  `(prompt_tokens - cached_tokens)*150 + cached_tokens*30 + completion_tokens*500`
- Billing is **prepaid**.
  - The maximum possible cost is reserved before generation. Without enough balance the call fails with `429 insufficient_quota`.
  - Always set `max_completion_tokens` to keep the reservation small.
  - `GET /v1/usage` (with a key) returns `available_nanodollars` and `reserved_nanodollars` for the current UTC month.
- I found no mention of free credits. **Any spend needs the user's approval (CLAUDE.md rule 5).**

### 2.2 Usage snippet

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://api.paperinstruments.com/v1",
                api_key=os.environ["PAPERINSTRUMENTS_API_KEY"])
resp = client.chat.completions.create(
    model="ultramarine-1-flash",
    messages=[{"role": "user", "content": "..."}],
    reasoning_effort="low",          # low | high | max (default max)
    max_completion_tokens=2048,      # includes reasoning; bounds prepaid reservation
    temperature=1, seed=0,
)
u = resp.usage
cached = (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0
reasoning = (u.completion_tokens_details.reasoning_tokens if u.completion_tokens_details else 0) or 0
```
The openai SDK version needed for `reasoning_effort="max"` is **UNVERIFIED**. Older SDKs type-hint `low/medium/high`, but they pass the string through unchanged. If the SDK rejects it, use `extra_body`.
