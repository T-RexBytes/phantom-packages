# API notes: Ollama, PyPI, pypistats, top-pypi-packages, Claude Code hooks, GitHub Actions

Checked on 2026-10-28. Every claim has a source. "LIVE" means I saw it in a real response on 2026-10-28. "UNVERIFIED" means I could not confirm it.

---

## 1. Ollama HTTP API

**Authoritative source:** https://docs.ollama.com/api/chat. The older `docs/api.md` on GitHub (https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md) is **stale** and does not mention `logprobs`, so don't use it.

Local server (LIVE): `GET /api/version` returns `{"version":"0.40.11"}`. The latest GitHub release is `v0.40.12` (2026-10-27, https://api.github.com/repos/ollama/ollama/releases/latest).

### POST /api/chat: request (https://docs.ollama.com/api/chat)

| field | notes |
|---|---|
| `model` | required |
| `messages` | required. `[{role: system\|user\|assistant\|tool, content, images?, tool_calls?, thinking?, tool_name?}]` |
| `stream` | bool, **default `true`** (NDJSON, `application/x-ndjson`). Send `false` to get one object back. |
| `think` | `true` / `false` / `null` (model default) / a string level. Numbers are not allowed. |
| `options` | `seed`, `temperature`, `top_k`, `top_p`, `min_p`, `stop`, `num_ctx`, `num_predict` (max tokens to generate) |
| `format` | `"json"` or a JSON Schema object |
| `keep_alive` | e.g. `"5m"`. `0` unloads the model. |
| `logprobs` | bool. Returns logprobs for output tokens. |
| `top_logprobs` | int. Number of alternatives per position (needs `logprobs: true`). The docs state no maximum (UNVERIFIED cap). |
| `tools` | function tool definitions |

```json
{
  "model": "qwen3:4b",
  "messages": [{"role": "user", "content": "..."}],
  "stream": false,
  "think": false,
  "options": {"temperature": 0.7, "seed": 42, "num_predict": 512},
  "logprobs": true,
  "top_logprobs": 5
}
```

### Thinking (https://docs.ollama.com/capabilities/thinking, https://docs.ollama.com/api-reference/show-model-details.md)
- **Disable thinking with `"think": false`.** The docs say it means "no thinking output, *if the model permits it*". `null` or omitting it uses the model default.
- String levels are model-defined. gpt-oss advertises `["low","medium","high"]` (default `"medium"`). `/api/chat` also mentions `"max"`. A level the model doesn't support falls back to the model default.
- To find a model's supported values, call `POST /api/show` and read `thinking.values` / `thinking.default`. The docs example for `gemma4` is `values: [false, true]`, `default: true`. **Gemma4 thinks by default**, so the project must send `think: false` explicitly. An array containing only `[false]` means the model has no thinking support. If `thinking` is absent, the model "might conduct thinking based on its existing behavior".
- Thinking text is kept separate from the answer:
  - `/api/chat`: `message.thinking` (trace) and `message.content` (answer)
  - `/api/generate`: `thinking` and `response`
- Local models (LIVE `/api/tags`): `qwen3:4b`, `gemma4:e2b` and `gemma4:e4b` all list `"thinking"` in `capabilities`.
- UNVERIFIED: whether `think:false` fully suppresses thinking for qwen3:4b and gemma4 on 0.40.11. I didn't call `/api/show` or generate, per the task limits. Check this in the extractor tests by asserting that `message.thinking` is empty or absent, and strip any stray `<think>` text from `content` as a safeguard.

### Response (non-streaming) (https://docs.ollama.com/api/chat)
Fields: `model`, `created_at` (ISO 8601), `message{role:"assistant", content, thinking?, tool_calls?, images?}`, `done`, `done_reason` (e.g. `"stop"`), `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`, `logprobs?`.
- **All `*_duration` values are integer nanoseconds.** For example, `total_duration: 174560334` is about 0.17 s.
- `prompt_eval_count` is the number of prompt tokens. `eval_count` is the number of generated tokens.

### Logprobs: AVAILABLE (since v0.12.11, 2025-11-12)
- Release note: https://github.com/ollama/ollama/releases/tag/v0.12.11. It says "Ollama's API and OpenAI-compatible API now support log probabilities".
- Both `/api/chat` and `/api/generate` accept `logprobs` and `top_logprobs` (https://docs.ollama.com/api/chat, https://docs.ollama.com/api/generate.md).
- The response is a top-level `logprobs` array (not inside `message`). Each entry has this shape:
```json
{"token": "Hello", "logprob": -0.836, "bytes": [72,101,108,108,111],
 "top_logprobs": [{"token": "Hello", "logprob": -0.836, "bytes": [...]},
                  {"token": "Hi", "logprob": -1.259, "bytes": [...]}]}
```
- The `/api/generate` docs list `logprobs` on the non-streaming `GenerateResponse` but not on the stream-event schema. The v0.12.11 notes say stream chunks include it. Use `stream:false` to be safe.
- UNVERIFIED: whether logprobs cover thinking tokens, and whether any backend runner (local `gemma4` shows `runner: "llamacpp"` and `qwen3` shows `"ggml"`) omits them. Test this once before relying on logprob features.

### GET /api/tags (https://docs.ollama.com/api/tags.md)
`{"models":[{name, model, modified_at, size (bytes), digest (sha256), details{format, family, families, parameter_size, quantization_level, parent_model}, remote_model?, remote_host?}]}`

LIVE extras on 0.40.11 that the docs don't list: `details.context_length`, `details.embedding_length`, `details.runner`, and a top-level `capabilities` array. Example (LIVE):
```json
{"name":"qwen3:4b","model":"qwen3:4b","modified_at":"2026-10-22T17:09:42.2877507+05:30","size":2497293931,
 "digest":"359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
 "details":{"parent_model":"","format":"gguf","family":"qwen3","families":["qwen3"],"parameter_size":"4.0B",
            "quantization_level":"Q4_K_M","context_length":262144,"embedding_length":2560,"runner":"ggml"},
 "capabilities":["completion","tools","thinking"]}
```
Also installed (LIVE): `gemma4:e2b` (4.6B, Q4_K_M, digest `b37049369adf…`) and `gemma4:e4b` (7.5B, Q4_K_M, digest `dc35e8d9c606…`). Record the `digest` in `results/models.json` for reproducibility.

### POST /api/show (https://docs.ollama.com/api-reference/show-model-details.md)
Request `{"model": "...", "verbose": false}`. Response fields: `capabilities[]`, `details{}`, `model_info{}` (e.g. `general.parameter_count`), `modified_at`, `parameters` (text), `template`, `thinking{values[], default}` (optional), plus `modelfile` (per api.md). There is no "release date" field. Use `modified_at` (local pull time) and `digest` instead.

---

## 2. PyPI JSON API (https://docs.pypi.org/api/json/)

- `GET https://pypi.org/pypi/{project}/json` returns these top-level keys (LIVE for `requests`): `info`, `last_serial`, `ownership`, `releases`, `urls`, `vulnerabilities`.
- `GET https://pypi.org/pypi/{project}/{version}/json` returns the same structure **without** `releases`.
- `info.*` (LIVE): author, author_email, bugtrack_url, classifiers, description, description_content_type, docs_url, download_url, downloads, dynamic, home_page, keywords, license, license_expression, license_files, maintainer, maintainer_email, name, package_url, platform, project_url, project_urls, provides_extra, release_url, requires_dist, requires_python, summary, version, yanked, yanked_reason.
- `releases`: `{version: [file, ...]}`. **Deprecated** on the project endpoint: "projects should shift to using the Index API to get this information, where possible". The docs also warn "In the future, each of these keys may be removed entirely". They give no date. As of 2026-10-28 it is still **present** (LIVE: 163 versions for `requests`). Other deprecated keys: `downloads` (always `-1`), `has_sig`, `bugtrack_url`.
- File object (LIVE): `comment_text, core-metadata, digests{blake2b_256,md5,sha256}, downloads(-1), filename, has_sig, md5_digest, packagetype, python_version, requires_python, size, upload_time, upload_time_iso_8601, url, yanked, yanked_reason`.
- `urls`: the file objects for the latest release.
- `ownership`: `{roles:[{role,user}], organization}`.
- **Missing project: HTTP 404** (LIVE).

### Alternative: Simple / Index API JSON (PEP 691/700) (https://docs.pypi.org/api/index-api/)
- `GET https://pypi.org/simple/{project}/` with header `Accept: application/vnd.pypi.simple.v1+json`. The response Content-Type is the same (LIVE).
- Top-level keys (LIVE): `meta{_last-serial, api-version:"1.4"}`, `name`, `versions[]`, `files[]`, `project-status{status}`.
- File entry (LIVE):
```json
{"core-metadata": false, "data-dist-info-metadata": false, "filename": "requests-2.34.2.tar.gz",
 "hashes": {"sha256": "f28892..."}, "provenance": "https://pypi.org/integrity/requests/2.34.2/requests-2.34.2.tar.gz/provenance",
 "requires-python": ">=3.10", "size": 142856, "upload-time": "2026-05-14T19:25:27.735762Z",
 "url": "https://files.pythonhosted.org/...", "yanked": false}
```
- Missing project: HTTP 404 (LIVE). `yanked` can be `false` or a reason string, per PEP 592/691.
- Recommendation: use `/pypi/{name}/json` for `info` (summary, project_urls, etc.). Use `/simple/` JSON for first and last upload time and version count if `releases` ever disappears. Both are metadata only.

### Rate limits and User-Agent (https://docs.pypi.org/api/)
- "there is currently no rate limiting of PyPI APIs at the edge". Also: "Try not to make a lot of requests (thousands) in a short amount of time (minutes)", "it's preferred to make requests in serial over a longer amount of time". PyPI may ban irresponsible consumers. Our 5 req/s cap is compatible with this guidance.
- "Set your consumer's `User-Agent` header to uniquely identify your requests. Adding your contact information ... would be helpful."
- Responses are CDN-cached (`X-Cache`, `X-Cache-Hits`) and include an `ETag`. Respect it with `If-None-Match`.
- XML-RPC: don't use it ("planned for deprecation").

---

## 3. pypistats.org API (https://pypistats.org/api/)

- `GET https://pypistats.org/api/packages/{name}/recent`, optional `?period=day|week|month`.
- LIVE response for `requests`:
```json
{"data":{"last_day":55719098,"last_month":1202777296,"last_week":330570977},"package":"requests","type":"recent_downloads"}
```
- **Unknown package: HTTP 404** (LIVE).
- **Name must be lowercase/normalised**: `/api/packages/Requests/recent` returned **404** (LIVE). Normalise with PEP 503 (`re.sub(r"[-_.]+","-",name).lower()`) before calling. UNVERIFIED: whether `_` and `.` variants also 404 (`-` is the safe form).
- Rate limit: the docs say only "IP-based rate limiting is imposed application-wide". LIVE headers: `X-Ratelimit-Limit: 60`, `X-Ratelimit-Remaining`, `X-Ratelimit-Reset` (epoch seconds), `Retry-After`. The window length is UNVERIFIED (60 per minute is likely), so throttle to at most 1 req/s here and back off on 429.
- Data is updated once daily. Time series are kept for 180 days. Mirrors are excluded by default. The docs ask users not to bulk-download all packages (use BigQuery for that) and to cache results for a day.
- Other endpoints: `overall`, `python_major`, `python_minor`, `system`.

---

## 4. Top PyPI packages (hugovk) (https://github.com/hugovk/top-pypi-packages)

- **Canonical URL:** `https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json` (unminified: `.../top-pypi-packages.json`). The old `hugovk.github.io/top-pypi-packages/...` URL **redirects** there (LIVE). A `top-pypi-packages-30-days.min.json` also exists and was byte-identical on 2026-10-28 (787,401 bytes). The 365-day files were removed (last updated 2021-04-01).
- Shape (LIVE):
```json
{"last_update": "2026-10-01 12:40:51", "source": "ClickHouse",
 "meta": [{"name":"download_count","type":"Int64"},{"name":"project","type":"String"}],
 "rows": [{"download_count": 2460723068, "project": "boto3"}, {"download_count": 1550430820, "project": "packaging"}, "..."],
 "total_rows": 15000, "rows_before_limit_at_least": 931904, "statistics": {"...": "..."}}
```
- **Row count: 15,000** (LIVE `len(rows)`). The last row has about 53k downloads per 30 days. The list is updated monthly (on the 1st, via GitHub Actions). The source is the ClickHouse PyPI dataset (clickpy).
- License: **UNVERIFIED**. The GitHub page shows no LICENSE file and no license in the sidebar. The README links a Zenodo record, so check that for a data license before redistributing. Fetching the file at runtime, caching it under `data/cache/` and citing it is the safe route.

---

## 5. Claude Code hooks (https://code.claude.com/docs/en/hooks)

### PreToolUse stdin JSON
```json
{
  "session_id": "abc123",
  "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
  "transcript_path": "/home/user/.claude/projects/.../transcript.jsonl",
  "cwd": "/home/user/my-project",
  "scratchpad_dir": "/tmp/claude-1000/-home-user-my-project/abc123/scratchpad",
  "permission_mode": "default",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {"command": "npm test", "description": "Run test suite", "timeout": 120000, "run_in_background": false},
  "tool_use_id": "toolu_01ABC123..."
}
```
- Common fields: `session_id`, `transcript_path`, `cwd`, `permission_mode` (`default|plan|acceptEdits|auto|dontAsk|bypassPermissions`), `hook_event_name`. Optional: `prompt_id`, `scratchpad_dir`, `effort{level}`, `agent_id`/`agent_type`.
- **The PowerShell tool has the same `tool_input` shape** (`command`, `description`, `timeout`, `run_in_background`). The docs recommend matcher `"Bash|PowerShell"` to cover both shell tools. **On Windows this matters: a `"Bash"`-only matcher misses PowerShell tool calls.**

### Exit codes
| code | effect |
|---|---|
| 0 | Success. JSON on stdout is applied. Plain stdout goes to the debug log. Claude never sees stderr from an exit-0 hook. |
| **2** | **Block.** On PreToolUse the call is blocked, and **stderr is the reason shown to Claude** (as `PreToolUse:Bash hook error: [...]: <stderr>`). JSON can't override the block. |
| other (e.g. 1) | **Non-blocking** error. The tool still runs. Exit 1 is NOT a block. |

### JSON-output alternative (exit 0, print only JSON)
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse",
  "permissionDecision": "deny",
  "permissionDecisionReason": "Blocked: installing model-proposed package"}}
```
- `permissionDecision`: `allow` (skips the prompt, but deny and ask rules are still evaluated), `deny` (blocks; the reason is shown to Claude), `ask` (the reason appears in the user prompt), `defer` (only with `-p`, non-interactive).
- When multiple hooks disagree, precedence is `deny > defer > ask > allow`.
- Also available: `updatedInput`, `additionalContext`, and the universal `continue`, `stopReason`, `systemMessage`, `suppressOutput` (no effect).
- Invalid JSON on exit 0 counts as a non-blocking error, so the tool runs. A guard should **prefer exit 2 plus stderr**, or make sure its JSON is valid.

### settings.json
```json
{"hooks": {"PreToolUse": [
  {"matcher": "Bash|PowerShell",
   "hooks": [{"type": "command",
              "command": "uv",
              "args": ["run", "--project", "${CLAUDE_PROJECT_DIR}", "python", "${CLAUDE_PROJECT_DIR}/hooks/phantom_guard.py"],
              "timeout": 30}]}]}}
```
(The structure above follows the docs. The exact command line is our proposal and UNVERIFIED in practice.)
- `matcher`: plain names or `|` lists match exactly. Anything else is treated as an unanchored JS regex. `"*"`, `""` or omitting it matches all tools.
- `type`: `command|http|mcp_tool|prompt|agent`. Optional `if` takes permission-rule syntax, e.g. `"Bash(pip *)"`.
- **`timeout` is in seconds. The default for command hooks is 600.** On timeout, a PreToolUse hook **does not block** unless `"onFailure": "block"` is set (v2.1.295+).
- `command` runs in shell form by default. If `args` is set, it runs in exec form (no shell). The `shell` field is `"bash"` or `"powershell"`.

### Windows notes
- Command hooks run in **Git Bash** on Windows, or **PowerShell** if Git Bash isn't installed. You can force this with `"shell": "powershell"`. Point to Git Bash with `CLAUDE_CODE_GIT_BASH_PATH` if needed.
- Without Git Bash, a PreToolUse exit code other than 0 or 2 counts as a failure and may block.
- Exec form needs a real `.exe` (`.cmd`/`.bat` shims can't be spawned). `uv.exe` and `python.exe` are fine.
- Quoting: in shell form, wrap path placeholders in double quotes. **Exec form (`args`) is recommended for any hook that uses a path placeholder.**
- `${CLAUDE_PROJECT_DIR}` is the project root where the session started. It is also exported as an env var to the hook process, and it doesn't change in worktrees.

---

## 6. GitHub Actions versions (as of 2026-10-28)

Source: GitHub REST `releases/latest` and `tags` (https://api.github.com/repos/{owner}/{repo}/releases/latest).

| action | latest release | floating major tag? | commit SHA of latest |
|---|---|---|---|
| `actions/checkout` | **v7.0.1** (2026-07-20) | yes, `@v7` | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `actions/setup-python` | **v7.1.0** (2026-10-22) | yes, `@v7` | `6d4fd22b20465ccc0f7354476588e8df7c9ca0e0` |
| `astral-sh/setup-uv` | **v10.3.3** (2026-10-28) | **NO.** Since v8.0.0 (2026-03-29) there are no major or minor tags, so `@v10` returns 404. Use `@v10.3.3` or the SHA. | `68559e99cc807c1eb7e9238fd43a15bba460eec2` |

- setup-uv v8.0.0 notes: "You won't be able to use `@v8` or `@v8.0` any longer" (https://github.com/astral-sh/setup-uv/releases/tag/v8.0.0).
- setup-uv v10.0.0 (2026-08-12) changed `enable-cache: auto` to disable the cache on `pull_request_target`, `workflow_run` and `release`, and added `version: "latest-known"` (https://github.com/astral-sh/setup-uv/releases/tag/v10.0.0).
- With setup-uv you usually don't need setup-python. `uv python install` or `python-version:` handles it.
