# AGENTS GUIDE — HK Legal Chatbot

Use executable sources first. If docs disagree with code/config, trust code/config.

## Source of truth (priority)
1. Runtime/config: `backend/**/*.py`, `backend/requirements.txt`, `frontend/package.json`, `frontend/src/**`
2. Sub-guides: `backend/AGENTS.md`, `frontend/AGENTS.md`
3. This file

## Real entrypoints and package boundaries
- Backend entry: `backend/main.py` (`if __name__ == "__main__": uvicorn.run(...)`).
- Frontend entry: `frontend/src/main.jsx` (`createRoot(...).render(...)`).
- API surface is only in `backend/main.py`: `GET /`, streaming `POST /chat`.
- Retrieval core lives in `backend/services/vector_store.py` (`VectorStoreManager`).
- Ollama failover/runtime probing lives in `backend/core/ollama_runtime.py`.
- Frontend stream parse/session logic lives in `frontend/src/components/ChatInterface.jsx`.

Prefer `backend/core/*` + `backend/services/*` over root-level duplicates (`backend/vector_store.py`, `backend/utils.py`, `backend/setup_env.py`, `backend/embedding_shared.py` are not the primary path used by `backend/main.py`).

## Exact developer commands

### Frontend (`workdir=frontend/`)
```bash
npm install
npm run dev
npm run lint
npm run build
npm run preview
```
`frontend/package.json` has no `test` or `typecheck` script.

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

Evaluation wrappers: `backend/scripts/run_eval_type1.ps1` and `backend/scripts/run_eval_type1.sh`.

## Critical invariants (easy to break)
1) **CUDA/TensorRT setup order is mandatory**
- Call `setup_env.setup_cuda_dlls()` before importing `torch`/`onnxruntime`.
- This order is used in `backend/main.py`, `backend/llm_evaluate.py`, `backend/services/*`, and key scripts.

2) **Do not mix Windows and WSL virtualenvs**
- Windows runtime: `.venv`; WSL/Linux runtime: `.venv-wsl`.
- `backend/requirements.txt` is uv-compiled with platform markers (Windows CUDA wheels vs non-Windows torch).

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

## Known stale script
- `backend/scripts/verify_trt.py` is currently stale (uses `time` without import and imports `BoostedYuanEmbeddings`, which is not in active services). Do not use it as a primary verification path until fixed.

## What to run after edits
- Frontend-only changes: `npm run lint` (plus `npm run build` for UI/build-sensitive edits).
- Backend-only changes: run touched script(s) + targeted `backend/tests/test_*.py` scripts.
- Stream/protocol changes: run `python backend\main.py` and verify parser behavior in `frontend/src/components/ChatInterface.jsx`.
