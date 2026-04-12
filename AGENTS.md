# AGENTS GUIDE — HK Legal Chatbot

Use executable sources first. If docs disagree with code/config, trust code/config.

## Source of truth (priority)
1. Runtime/config: `backend/**/*.py`, `backend/requirements.txt`, `frontend/package.json`, `frontend/src/**`
2. Sub-guides: `backend/AGENTS.md`, `frontend/AGENTS.md`
3. This file

## Real entrypoints and boundaries
- Backend app entry: `backend/main.py` (`uvicorn.run(...)` under `if __name__ == "__main__"`).
- Frontend app entry: `frontend/src/main.jsx` (`createRoot(...).render(...)`).
- API surface: `GET /` and streaming `POST /chat` in `backend/main.py`.
- Retrieval core: `backend/services/vector_store.py` (`VectorStoreManager`).
- Ollama connectivity/fallback logic: `backend/core/ollama_runtime.py`.
- Frontend stream parser + session persistence: `frontend/src/components/ChatInterface.jsx`.

## Exact developer commands

### Frontend (`workdir=frontend/`)
```bash
npm install
npm run dev
npm run lint
npm run build
npm run preview
```
`frontend/package.json` has no `test`/`typecheck` script.

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

### Backend run/verification commands (`workdir=repo root`)
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

Evaluation wrappers:
- PowerShell: `backend/scripts/run_eval_type1.ps1`
- Bash: `backend/scripts/run_eval_type1.sh`

## Critical invariants (easy to break)
1) **CUDA/TensorRT setup order is mandatory**
- Call `setup_env.setup_cuda_dlls()` before importing `torch`/`onnxruntime`.
- Pattern is used in `backend/main.py`, `backend/llm_evaluate.py`, `backend/services/*`, and key ingestion/verification scripts.

2) **Do not mix virtualenvs between Windows and WSL**
- Windows daily runtime: `.venv`
- WSL/Linux runtime: `.venv-wsl`
- `backend/requirements.txt` is uv-compiled and uses platform markers (Windows CUDA wheel vs non-Windows torch wheel).

3) **Backend/Frontend SSE contract is strict**
- Backend sends `text/event-stream` from `POST /chat`.
- Request body must include `message`, `language`, `session_id`.
- `session_id="default"` is rejected by backend with an error event.
- Frontend parser only handles lines prefixed with `data: ` and expects payload keys: `answer`, `references`, `error`.
- Ordering matters: answer chunks stream first, references are emitted at the end.
- Frontend session key is fixed: `localStorage['hk-legal-chatbot-session-id']`.

4) **Prefer `backend/core/*` and `backend/services/*` over root-level duplicates**
- Root-level files like `backend/vector_store.py`, `backend/utils.py`, `backend/setup_env.py`, `backend/embedding_shared.py` exist but are not the primary path used by `backend/main.py`.

## Environment loading and keys
- Backend modules commonly call both `load_dotenv()` and an explicit `backend/.env` load; launch from repo root to avoid cwd surprises.
- Required for full chat/retrieval flow: `DEEPSEEK_API_KEY`, `PINECONE_API_KEY`.
- Common runtime toggles: `PINECONE_INDEX_NAME`, `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_HOST_GATEWAY`.

## High-signal gotcha
- `backend/scripts/verify_trt.py` is stale in current tree (missing `time` import and references `BoostedYuanEmbeddings` not present in active modules). Do not rely on it as a primary verification script until fixed.

## Verification expectations after edits
- Frontend-only changes: run `npm run lint` (and `npm run build` for UI/build-sensitive changes).
- Backend-only changes: run the touched script(s) plus targeted `backend/tests/test_*.py` script checks.
- Stream/protocol changes: verify backend stream behavior (`python backend\main.py`) and frontend stream parsing in `ChatInterface.jsx`.
