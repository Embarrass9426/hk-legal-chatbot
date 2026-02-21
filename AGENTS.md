# PROJECT KNOWLEDGE BASE

## OVERVIEW

HK Legal Chatbot: Full-stack RAG application for parsing, indexing, and querying Hong Kong legal documents (e-Legislation PDFs). FastAPI backend with ONNX/TensorRT embeddings + Pinecone vector store. React/Vite/Tailwind frontend with streaming chat UI.

## STRUCTURE

```
.
├── backend/                # Python — FastAPI server, PDF parsing, vector DB
│   ├── core/               # Shared utilities (setup_env, embedding_shared, utils)
│   ├── services/           # Business logic (vector_store, embedding_service)
│   ├── parsers/            # PDF parsing (pdf_parser.py — PDFLegalParserV2)
│   ├── scripts/            # CLI tools (ingest_pdfs, batch_download, cap_discovery)
│   ├── tests/              # Ad-hoc test scripts (NOT pytest — use __main__ guards)
│   ├── bin/                # Bundled Poppler binaries (gitignored)
│   ├── data/               # PDFs + parsed JSON (gitignored)
│   ├── models/             # ONNX/TRT embedding models (gitignored)
│   ├── main.py             # FastAPI app entry point
│   └── requirements.txt    # Pinned dependencies
├── frontend/               # JavaScript — React + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/     # ChatInterface.jsx, ReferenceCard.jsx
│   │   ├── App.jsx         # Root component (dark mode state)
│   │   ├── main.jsx        # Vite entry point
│   │   └── index.css       # Tailwind directives + prose overrides
│   ├── eslint.config.js    # ESLint 9 flat config
│   ├── tailwind.config.js  # darkMode: 'class'
│   └── package.json        # Scripts: dev, build, lint, preview
└── docs/                   # Project documentation
```

## COMMANDS

### Backend (run from project root)

```bash
# Start API server (port 8000)
python backend/main.py

# Ingest PDFs into Pinecone
python backend/scripts/ingest_pdfs.py --cap 282 599A
python backend/scripts/ingest_pdfs.py --force-parse --skip-upload

# Run individual test scripts (no pytest — each has __main__ guard)
python backend/tests/test_dll.py
python backend/tests/test_embedding_similarity.py

# Install dependencies
pip install -r backend/requirements.txt
```

### Frontend (run from `frontend/` directory)

```bash
npm install          # Install dependencies
npm run dev          # Vite dev server with HMR
npm run build        # Production build
npm run lint         # ESLint check
npm run preview      # Preview production build
```

### No formal test suite exists. Tests are standalone scripts with `if __name__ == "__main__"` guards.

## CODE STYLE — BACKEND (Python)

### Formatting & Linting
- **Ruff** is used (`.ruff_cache/` present) with default settings — no custom config file.
- 4-space indentation. Double quotes for strings (mostly).
- Max line length: follow ruff defaults (~88 chars, flexible).

### Naming
- `snake_case` for functions, variables, modules, file names.
- `PascalCase` for classes (`VectorStoreManager`, `PDFLegalParserV2`, `EmbeddingService`).
- `UPPER_SNAKE` for constants (`STOP_TOKEN`, `DROP_TAGS`, `KEEP_TAGS`).
- Prefix private methods with `_` (`_load_model`, `_embed_batch`, `_slugify`).

### Imports
Order: stdlib → third-party → local. Local imports use absolute paths from `backend.*`:
```python
import os
import json
from typing import List, Dict, Any

from fastapi import FastAPI
from pinecone import Pinecone
from dotenv import load_dotenv

from backend.core import setup_env
from backend.services.embedding_service import get_embedding_service
```

### Type Hints
- Used on public method signatures (return types + params).
- Not enforced everywhere — be consistent with surrounding code.
- Use `typing` imports (`List`, `Dict`, `Any`, `Optional`).

### Error Handling
- `try/except` with `print()` for logging (no structured logging library).
- Fallback patterns: return safe defaults on failure (e.g., `return user_query` on rewrite failure).
- Bare `except Exception` with descriptive print messages.
- Scripts use `traceback.print_exc()` for detailed error output.

### Patterns
- **Singleton**: `EmbeddingService` uses `__new__` + `_lock` for thread-safe singleton.
- **Queue workers**: `embedding_shared.py` defines shared `job_q`/`result_q` for inter-thread comms.
- **`__main__` guards**: Every script/module should be directly runnable for testing.
- **`setup_env.setup_cuda_dlls()`**: MUST call before importing torch/onnxruntime. Always at module top.
- **`.env` via python-dotenv**: API keys loaded with `load_dotenv()`. Never hardcode keys.
- **Pydantic models**: Used for FastAPI request/response schemas (`ChatRequest`).

### Path Handling
- Use `os.path` relative to `__file__` — never hardcode absolute paths in committed code.
- `__main__` blocks may use absolute paths for local dev (acceptable, won't run in prod).

## CODE STYLE — FRONTEND (JavaScript/JSX)

### Formatting & Linting
- **ESLint 9** flat config with `react-hooks` + `react-refresh` plugins.
- Rule: `no-unused-vars` errors except variables matching `^[A-Z_]`.
- No Prettier config — follow existing formatting (2-space indent, single quotes for JS).

### Naming
- `PascalCase` for components and component files (`ChatInterface.jsx`, `ReferenceCard.jsx`).
- `camelCase` for variables, functions, props, state (`darkMode`, `toggleDarkMode`, `handleSend`).

### Components
- **Functional components only** with hooks (`useState`, `useEffect`).
- Props via destructuring: `const ChatInterface = ({ darkMode, toggleDarkMode }) => { ... }`.
- Default export per component file.
- No TypeScript — plain `.jsx`.

### Styling
- **Tailwind CSS 4.0** utility classes exclusively. No component-level CSS files.
- Dark mode via `dark:` variant (class-based: `darkMode: 'class'` in tailwind config).
- Global prose styles in `src/index.css` using `@apply`.
- Color palette: `slate-*` for neutrals, `blue-*` for primary actions.

### State & API
- Local state via `useState`/`useEffect`. No Redux/Context.
- Direct `fetch()` to `http://localhost:8000`. SSE streaming via `ReadableStream` reader.

## WHERE TO LOOK

| Task                    | Location                              | Notes                                            |
|-------------------------|---------------------------------------|--------------------------------------------------|
| API endpoints           | `backend/main.py`                     | FastAPI routes: `/chat` (POST), `/` (GET)        |
| PDF parsing             | `backend/parsers/pdf_parser.py`       | `PDFLegalParserV2` — unstructured-based parser   |
| Vector search           | `backend/services/vector_store.py`    | `VectorStoreManager` — Pinecone + embeddings     |
| Embedding model         | `backend/services/embedding_service.py`| Singleton ONNX/TRT service with mean pooling    |
| Query rewriting         | `backend/core/utils.py`              | LLM-powered query rewrite for retrieval          |
| CUDA/DLL setup          | `backend/core/setup_env.py`          | Must run before torch/onnx imports               |
| PDF ingestion pipeline  | `backend/scripts/ingest_pdfs.py`     | CLI: parse → embed → upsert to Pinecone          |
| Chat UI                 | `frontend/src/components/ChatInterface.jsx` | Streaming chat with SSE                    |
| Reference cards         | `frontend/src/components/ReferenceCard.jsx` | Legal citation display                     |
| App shell + dark mode   | `frontend/src/App.jsx`               | Root component, dark mode toggle                 |

## ENVIRONMENT

- **Python**: 3.12+
- **Node**: Compatible with Vite 7.x / React 19.x
- **GPU**: CUDA 12.4 + TensorRT optional (falls back to CPU)
- **External services**: Pinecone (vector DB), DeepSeek API (LLM)
- **API keys**: Stored in `backend/.env` (gitignored). Required: `DEEPSEEK_API_KEY`, `PINECONE_API_KEY`

## ANTI-PATTERNS — DO NOT

- Import torch/onnxruntime before calling `setup_env.setup_cuda_dlls()`.
- Hardcode absolute paths in committed code (only allowed in `__main__` blocks).
- Add CSS files per component — use Tailwind utilities.
- Commit `.env`, PDFs, model binaries, or `backend/bin/` contents.
- Suppress errors with empty `except: pass` — always log or handle.
