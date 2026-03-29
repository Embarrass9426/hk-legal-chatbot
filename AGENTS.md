# PROJECT KNOWLEDGE BASE

## PURPOSE

Root reference for agents working in HK Legal Chatbot: a FastAPI + ONNX/TensorRT backend, Pinecone retrieval, and a React/Vite/Tailwind frontend for streaming legal Q&A over Hong Kong e-Legislation PDFs.

## REPO MAP

- `backend/`: Python app, parsers, services, scripts, and standalone test scripts.
- `frontend/`: React + Vite + Tailwind UI with chat and reference components.
- `docs/`: design notes and parser/RAG writeups.
- `backend/bin/`, `backend/data/`, `backend/models/`: gitignored runtime assets.
- `backend/main.py`: FastAPI app entry point.
- `backend/core/setup_env.py`: CUDA/DLL bootstrap before `torch` or `onnxruntime`.
- `backend/core/utils.py`: retrieval rewrite helpers.
- `backend/services/embedding_service.py`: singleton embedding service.
- `backend/services/vector_store.py`: Pinecone/vector-store logic.
- `backend/parsers/pdf_parser.py`: `PDFLegalParserV2`.
- `backend/scripts/ingest_pdfs.py`: PDF ingest workflow.
- `backend/tests/`: standalone verification scripts, not pytest.
- `frontend/package.json`: `dev`, `build`, `lint`, `preview` scripts.
- `frontend/eslint.config.js`: ESLint 9 flat config.
- `frontend/tailwind.config.js`: class-based dark mode.
- `frontend/src/App.jsx`: app shell and dark-mode state.
- `frontend/src/components/ChatInterface.jsx`: streaming chat UI.
- `frontend/src/components/ReferenceCard.jsx`: citation cards.
- `frontend/src/index.css`: global prose styles.
- `docs/PDF_PARSER_LOGIC.md` and `docs/RAG_DESIGN.md`: design references.

## AGENTS HIERARCHY

`backend/AGENTS.md` and `frontend/AGENTS.md` override this root guidance inside their subtrees.

## EDITOR-RULE NOTE

No `.cursor/rules/**`, `.cursorrules`, or `.github/copilot-instructions.md` files exist here; follow AGENTS files only.

## COMMANDS

### Backend (run from repo root)

```bash
python backend/main.py
python backend/scripts/ingest_pdfs.py --cap 282 599A
python backend/scripts/ingest_pdfs.py --force-parse --skip-upload
uv pip install -r backend/requirements.txt
```

### Backend tests (standalone scripts; run from repo root)

```bash
python backend/tests/test_dll.py
python backend/tests/test_embedding_similarity.py
python backend/tests/test_tensorrt_embedding.py
```

These are the only test workflow entries in this repo; there is no pytest suite.

### Frontend (run from `frontend/`)

```bash
npm install
npm run dev
npm run build
npm run lint
npm run preview
```

## WORKFLOW NOTES

- Keep repo-wide guidance in this file; let subtree AGENTS files handle local overrides.
- Use the backend test scripts individually when checking runtime behavior.
- The frontend scripts are the only documented npm workflows here.
- Prefer the file paths listed below when orienting yourself in the codebase.
- Keep changes grounded in the files that actually exist in this repository.

## BACKEND STYLE

- Ruff default settings; 4-space indent; ~88 character line length.
- Naming: `snake_case` for functions/vars/modules, `PascalCase` for classes, `UPPER_SNAKE` for constants, `_prefix` for private methods.
- Imports: stdlib → third-party → local; local imports use `backend.*` absolute paths.
- `main.py` bare sibling imports (`import utils`) are a legacy exception; do not copy that pattern into new code.
- Call `setup_env.setup_cuda_dlls()` before any `torch` or `onnxruntime` import.
- Load API keys with `load_dotenv()` from `backend/.env`; never hardcode secrets.
- Public call signatures use type hints; prefer `typing` imports such as `List`, `Dict`, `Any`, `Optional`.
- Error handling uses `try/except Exception` with `print()`; scripts use `traceback.print_exc()`; never `except: pass`.
- Every script intended for direct execution should keep an `if __name__ == "__main__"` guard.
- Use `os.path` relative to `__file__`; avoid absolute paths in committed code.
- Pydantic models are used for request/response schemas in the FastAPI app.
- Follow the existing `print()`-based error reporting style in backend modules and scripts.
- Keep direct-run scripts import-safe and guarded by `if __name__ == "__main__"`.

## FRONTEND STYLE

- ESLint 9 flat config in `frontend/eslint.config.js`; `react-hooks` and `react-refresh` plugins are enabled.
- `no-unused-vars` errors except names matching `^[A-Z_]`.
- 2-space indent and single quotes for JS strings.
- Functional components only, with hooks; destructure props; default export per `.jsx` file.
- Tailwind CSS 4.0 utility classes only; no per-component CSS files.
- Dark mode is class-based with `darkMode: 'class'` in `tailwind.config.js`.
- Global prose styles live in `frontend/src/index.css`; `@tailwindcss/typography` is used; `lucide-react` supplies icons.
- State stays local with `useState` and `useEffect`; no Redux or Context.
- Markdown rendering uses `react-markdown` plus `remark-gfm`.
- Chat requests use direct `fetch()` to `http://localhost:8000`, with SSE streaming via `ReadableStream`.
- The app uses functional components only; no class components are present.
- `lucide-react` supplies icons in the UI.
- Tailwind utility classes should cover styling needs without component CSS files.

## WHERE TO LOOK

| Task | Location | Notes |
|---|---|---|
| API endpoints | `backend/main.py` | `/chat` streaming POST, `/` health |
| PDF parsing | `backend/parsers/pdf_parser.py` | `PDFLegalParserV2` |
| Vector search | `backend/services/vector_store.py` | `VectorStoreManager` |
| Embeddings | `backend/services/embedding_service.py` | Singleton ONNX/TRT service |
| Query rewrite | `backend/core/utils.py` | Retrieval rewrite helpers |
| Env bootstrap | `backend/core/setup_env.py` | Load DLLs before torch/onnxruntime |
| CUDA/DLL setup | `backend/core/setup_env.py` | CUDA DLL prep before imports |
| PDF ingestion | `backend/scripts/ingest_pdfs.py` | Parse, embed, upsert flow |
| Backend tests | `backend/tests/` | `test_dll.py`, `test_embedding_similarity.py`, `test_tensorrt_embedding.py` |
| Chat UI | `frontend/src/components/ChatInterface.jsx` | Streaming chat UI |
| Reference cards | `frontend/src/components/ReferenceCard.jsx` | Citation display |
| App shell | `frontend/src/App.jsx` | Root UI and dark mode |
| Global styles | `frontend/src/index.css` | Prose styles and Tailwind base |
| ESLint config | `frontend/eslint.config.js` | Flat config and `no-unused-vars` rule |
| Tailwind config | `frontend/tailwind.config.js` | `darkMode: 'class'` |

## OPERATING CONSTRAINTS

- No pytest, no `npm test`, and no extra test runner assumptions.
- No speculative tooling or commands; use only the scripts documented above.
- No TypeScript rules: this repo is JavaScript/JSX only.
- No `.env`, PDF, model, or `backend/bin/` content should be treated as source files.
- No per-component CSS files should be introduced for UI work.
- No structured-logging requirement is documented here; backend code uses prints and tracebacks.
- No new npm dependency should be added without checking existing packages first.

## ENVIRONMENT

- Python 3.12+; `uv` preferred, pip is the fallback.
- Node targets Vite 7.x / React 19.x and uses ESM.
- CUDA 12.4 + TensorRT are optional; CPU/ONNX fallback exists.
- External services: Pinecone and DeepSeek via OpenAI-compatible API.
- Required API keys live in `backend/.env`: `DEEPSEEK_API_KEY` and `PINECONE_API_KEY`.
- Runtime artifacts under `backend/bin/`, `backend/data/`, and `backend/models/` are gitignored.

## ANTI-PATTERNS

- Never import `torch` or `onnxruntime` before `setup_env.setup_cuda_dlls()`.
- Never commit absolute paths except in `__main__` blocks.
- Never add per-component CSS files; use Tailwind utilities only.
- Never commit `.env`, PDFs, model binaries, or `backend/bin/` contents.
- Never use `except: pass`; always log or surface the error.
- Never use TypeScript suppression patterns in this JS project.
- Never add new npm dependencies without first checking `lucide-react` and existing dependencies.
- Never rely on Cursor or Copilot config files here; none exist in the repo.
