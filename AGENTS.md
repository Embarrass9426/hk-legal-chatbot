# AGENTS GUIDE — HK Legal Chatbot

Use executable sources first. If docs disagree with code/config, trust code/config.

## Source of truth (priority)
1. Runtime/config: `backend/**/*.py`, `backend/requirements.txt`, `frontend/package.json`, `frontend/src/**`
2. Environment template: `backend/.env.example`
3. Sub-guides: `backend/AGENTS.md`, `frontend/AGENTS.md`
4. This file

## Real entrypoints and package boundaries
- Backend entry: `backend/main.py` (`if __name__ == "__main__": uvicorn.run(...)`).
- Frontend entry: `frontend/src/main.jsx` (`createRoot(...).render(...)`).
- API surface is only in `backend/main.py`: `GET /`, streaming `POST /chat`.
- Retrieval core lives in `backend/services/vector_store.py` (`VectorStoreManager`).
- Ollama failover/runtime probing lives in `backend/core/ollama_runtime.py`.
- Frontend stream parse/session logic lives in `frontend/src/components/ChatInterface.jsx`.

Prefer `backend/core/*` + `backend/services/*` over root-level duplicates (`backend/vector_store.py`, `backend/utils.py`, `backend/setup_env.py`, `backend/embedding_shared.py` are stale and not imported by `main.py`).

## Exact developer commands

### Frontend (`workdir=frontend/`)
```bash
npm install
npm run dev
npm run lint
npm run build
npm run preview
```
- `frontend/package.json` has no `test` or `typecheck` script.
- Tailwind CSS 4.0, React 19.2, ESLint 9 flat config (`eslint.config.js`).
- Dark mode via `darkMode: 'class'` in `tailwind.config.js`.

### Backend setup (`workdir=repo root`)
Windows:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

WSL/Linux:
```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```
- Requirements are uv-compiled with platform markers (Windows CUDA wheels vs non-Windows torch). Do not mix Windows and WSL virtualenvs.

### Backend run and focused verification (`workdir=repo root`)
```powershell
python backend\main.py
python backend\llm_evaluate.py
python backend\scripts\ingest_pdfs.py --cap 282 599A
python backend\scripts\verify_cpu_rerank_pipeline.py --query "<query>" --mode fast
python backend\scripts\ollama_discover.py
python backend\scripts\ollama_lifecycle.py status
python backend\tests\test_dll.py
python backend\tests\test_embedding_similarity.py
python backend\tests\test_tensorrt_embedding.py
python backend\tests\test_reranker_tokenizer_unicode.py
```
- `backend/tests/` are ad-hoc scripts with `__main__` guards, NOT pytest.
- Evaluation wrappers: `backend/scripts/run_eval_type1.ps1` and `backend/scripts/run_eval_type1.sh`.

## Critical invariants (easy to break)
1) **CUDA/TensorRT setup order is mandatory**
- Call `setup_env.setup_cuda_dlls()` before importing `torch`/`onnxruntime`.
- This order is used in `backend/main.py`, `backend/llm_evaluate.py`, `backend/services/*`, and key scripts.

2) **Do not mix Windows and WSL virtualenvs**
- Windows runtime: `.venv`; WSL/Linux runtime: `.venv-wsl`.

3) **SSE contract between backend and frontend is strict**
- Backend returns `text/event-stream` from `POST /chat`.
- Request JSON must include `message`, `language`, `session_id`.
- `session_id="default"` is rejected with an SSE `error` payload.
- Frontend only parses lines starting with `data: ` and expects payload keys `answer`, `references`, `error`.
- Stream order matters: answer chunks first, references near stream end.
- Session storage key is fixed: `localStorage['hk-legal-chatbot-session-id']`.

## Environment + runtime toggles that matter
- Backend commonly does both `load_dotenv()` and explicit `backend/.env` loading; run from repo root to avoid cwd surprises.
- Required for full chat/retrieval flow: `DEEPSEEK_API_KEY`, `PINECONE_API_KEY`.
- Common toggles: `PINECONE_INDEX_NAME`, `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_HOST_GATEWAY`.
- See `backend/.env.example` for the full set of env vars and defaults.

## Known stale scripts
- `backend/scripts/verify_trt.py` is currently stale (uses `time` without import and imports `BoostedYuanEmbeddings`, which is not in active services). Do not use it as a primary verification path until fixed.
- `backend/evaluate_models.py` imports from stale root-level modules (`vector_store`, `utils`) rather than `backend.services` / `backend.core`.

## What to run after edits
- Frontend-only changes: `npm run lint` (plus `npm run build` for UI/build-sensitive edits).
- Backend-only changes: run touched script(s) + targeted `backend/tests/test_*.py` scripts.
- Stream/protocol changes: run `python backend\main.py` and verify parser behavior in `frontend/src/components/ChatInterface.jsx`.

## Repo-specific notes
- `backend/bin/` contains bundled Poppler binaries required for PDF parsing.
- `backend/models/` holds local ONNX embedding models with optional TensorRT acceleration.
- Most backend modules can be run directly as scripts for testing (check `__main__` guards).
- `backend/main.py` uses bare imports (`import utils`) due to `sys.path` manipulation — do not follow this pattern in new code; prefer `backend.*` absolute imports.
