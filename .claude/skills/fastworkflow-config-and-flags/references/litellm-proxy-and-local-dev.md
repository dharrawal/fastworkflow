# LiteLLM Proxy recipe + repo-local dev env conventions

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Companion to
[../SKILL.md](../SKILL.md) — read that first for the precedence rules and the catalog.

## 1. LiteLLM Proxy routing recipe

A **LiteLLM Proxy** is a standalone gateway server (often fronting AWS Bedrock or a corporate
LLM gateway). fastWorkflow only ever acts as a *client*; the core dependency is plain `litellm`
(never `litellm[proxy]` — see the rationale comment block in `pyproject.toml` around lines
55-64). You do NOT need the `[server]` extra for proxy routing.

Routing logic lives entirely in `fastworkflow/utils/dspy_utils.py:42-69` (`get_lm`):

1. Read the model string from the role var. If it starts with `litellm_proxy/`:
2. Require `LITELLM_PROXY_API_BASE` — `ValueError` with a self-explanatory message if unset
   (`dspy_utils.py:50-55`).
3. Optionally read `LITELLM_PROXY_API_KEY` (`default=None`, so no-auth proxies work and a
   shell export IS honored for this one, `dspy_utils.py:58`).
4. The per-role `LITELLM_API_KEY_<ROLE>` is IGNORED for proxied models.

Copy-paste configuration (in `fastworkflow.env`):

```bash
LLM_AGENT=litellm_proxy/bedrock_mistral_large_2407
LITELLM_PROXY_API_BASE=http://127.0.0.1:4000
```

and in `fastworkflow.passwords.env` (only if your proxy requires auth):

```bash
LITELLM_PROXY_API_KEY=<proxy key>
```

### Proxy caveats (verified in code, labeled where inferred)

- **`LLM_PARAM_EXTRACTION` likely does not proxy-route.** `utils/signatures.py:255`
  instantiates `dspy.LM(LLM_PARAM_EXTRACTION, api_key=LITELLM_API_KEY_PARAM_EXTRACTION)`
  directly, bypassing `get_lm`, so no `api_base` is passed. Code-read inference: a
  `litellm_proxy/` model string for this role will fail or misroute. NOT runtime-verified.
  If you hit this, that call site is the fix target (route it through `get_lm`).
- **`LLM_SYNDATA_GEN` proxy behavior is unverified.** `train/generate_synthetic.py:35-36` and
  `utils/generate_param_examples.py:333-334` call `litellm.completion(model=..., api_key=...)`
  directly, not `get_lm`. Whether `litellm.completion` honors a `litellm_proxy/` prefix without
  an explicit `api_base` has not been tested here — treat as open.
- **`run` still demands `LITELLM_API_KEY_SYNDATA_GEN`** even in all-proxy setups
  (file-presence probe, `run/__main__.py:131-132`). Put a dummy value in the passwords file.
- **Bedrock note:** `train` only warns on the missing SYNDATA key ("OK if this is Bedrock",
  `train/__main__.py:276-277`); AWS credentials flow through boto3 (optional `aws` poetry
  group, `pyproject.toml`), outside the env-file system entirely.

## 2. Repo-local developer env conventions (this repo, not customer deployments)

These are conventions of the fastWorkflow repo itself; customer workflows only ever see the
two-file contract described in SKILL.md.

| Path | Tracked in git? | What it is |
|---|---|---|
| `env/.env` | YES (verified `git ls-files`) | Team model configuration for dev/tests — includes `LLM_COMMAND_METADATA_GEN`, which the packaged template omits |
| `passwords/.env` | no (local only) | Real API keys for dev/tests — includes `LITELLM_API_KEY_COMMANDMETADATA_GEN` |
| `.env` (repo root) | no (generated) | Output of `make gen-env` — merged from every `*.env` in the tree |
| `override/*.env` | optional | Processed last by gen-env.sh, so its values win |

- Tests initialize from the repo-local pair:
  `fastworkflow.init({**dotenv_values("./env/.env"), **dotenv_values("./passwords/.env")})`
  (`tests/test_command_executor.py:24-28`). FastAPI/MCP tests skip without them.
- `gen-env.sh` (invoked by `make gen-env`, `makefile:7-10`): finds every `*.env` excluding
  `./.env` and `./override/*` (plus `--exclude dir1,dir2`), later occurrences overwrite earlier,
  `override/` files processed last, output sorted into root `.env` (`gen-env.sh:89-116`).
  The generated root `.env` also carries `PYPI_ACCESS_TOKEN`/`TESTPYPI_ACCESS_TOKEN` for
  `make publish`.
- **Masking side effect (important):** importing `litellm` runs `load_dotenv()` over the CWD
  `.env` (`site-packages/litellm/__init__.py:29`). Because the repo root has that generated
  `.env`, code that wrongly reads env vars at import time (before `fastworkflow.init()`) works
  in-repo and breaks only in customer deployments. This masked the v2.21.0-2.21.3
  `SPEEDDICT_FOLDERNAME` import-time regression (fixed by lazy construction in 79e6986; see
  the comment at `fastworkflow/run_fastapi_mcp/utils.py:682-684`). When testing config
  changes for deployment realism, run from a directory WITHOUT a `.env`.
- The bundled example passwords template (`fastworkflow/examples/fastworkflow.passwords.env`)
  contains `<API KEY ...>` placeholders. `tests/test_train_modern_stack.py` guards against
  placeholders clobbering real keys with `_looks_like_real_key()` (rejects values containing
  `<` or `your-`).

## 3. Env-file-related discipline rules (confirmed with the owner, 2026-07-08/09)

- Never `git commit`/`push` config or template changes without the developer'sexplicit request in that
  turn (a private doc was once auto-pushed to the public repo, forcing a history rewrite).
- `passwords/.env`, root `.env`, and `jwt_keys/*.pem` contain secrets; they are untracked or
  gitignored — keep them that way. `env/.env` is tracked deliberately (models, no secrets);
  double-check nothing secret lands in it before any commit Dhar requests.

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

- Proxy routing logic: `sed -n '42,69p' fastworkflow/utils/dspy_utils.py`
- signatures.py bypass: `grep -n "dspy.LM(" fastworkflow/utils/signatures.py`
- litellm.completion call sites: `grep -rn "litellm.completion" fastworkflow/ --include="*.py"`
- Tracked-ness: `git ls-files env/.env passwords/.env .env`
- gen-env behavior: `sed -n '85,116p' gen-env.sh; grep -n "gen-env" makefile`
- litellm dotenv side effect: `grep -n "load_dotenv" .venv/lib/python*/site-packages/litellm/__init__.py`
- Test env pair: `sed -n '20,30p' tests/test_command_executor.py`
