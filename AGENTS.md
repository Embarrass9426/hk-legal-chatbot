# AGENTS GUIDE — HK Legal Chatbot
Purpose: practical instructions for coding agents operating in this repository.

Scope precedence:
1. Actual code behavior + tool output
2. Nearest subtree guide (`backend/AGENTS.md`, `frontend/AGENTS.md`)
3. This root `AGENTS.md`

---

## 1) Operating Model
- Monorepo with two active apps:
  - `backend/` — FastAPI backend + retrieval/embedding pipeline
  - `frontend/` — React + Vite + Tailwind UI
- Working directory rules:
  - Backend commands run from repo root (`backend\...` paths on Windows)
  - Frontend commands run inside `frontend/`
- Keep edits scoped to the target app unless cross-app behavior is required.

## 2) Cursor / Copilot Rules Status
Checked in repo root:
- `.cursorrules` → not found
- `.cursor/rules/` → not found
- `.github/copilot-instructions.md` → not found
No additional Cursor/Copilot instruction files currently apply.

---

## 3) Commands (Build / Lint / Test)

### 3.1 Frontend commands
Workdir: `frontend/`
Source: `frontend/package.json`
```bash
npm install
npm run dev
npm run build
npm run lint
npm run preview
```
Notes:
- `build` runs `vite build`
- `lint` runs `eslint .`
- No frontend `test` script is defined currently

### 3.2 Backend setup
Workdir: repo root
Windows setup (recommended):
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```
WSL/Linux setup (when intentionally using Linux runtime):
```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 3.3 Backend run commands
Workdir: repo root
```powershell
python backend\main.py
python backend\llm_evaluate.py
python backend\scripts\ingest_pdfs.py --cap 282 599A
```

### 3.4 Backend tests (single-test rule)
Workdir: repo root
Current tests are script-style files in `backend/tests/`.
Run one test file directly:
```powershell
python backend\tests\test_dll.py
python backend\tests\test_embedding_similarity.py
python backend\tests\test_tensorrt_embedding.py
```
Single-test rule:
- Execute one `test_*.py` file at a time: `python <path-to-test-file>`
- Do **not** assume `pytest -k`/marker workflows are configured

---

## 4) Code Style Guidelines

### 4.1 Cross-cutting
- Prefer small, local edits first; avoid broad refactors unless requested.
- Match surrounding style in each touched file.
- Reuse existing architecture/patterns before introducing abstractions.
- Never hardcode secrets; backend credentials belong in `.env`.

### 4.2 Python backend
Observed in `backend/main.py`, `backend/core/setup_env.py`, `backend/services/vector_store.py`.
- Imports follow: stdlib → third-party → local `backend.*`
- Naming: `snake_case` (functions/vars/modules), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)
- Typing: keep hints on public helpers and non-trivial internals
- Formatting: 4-space indentation, double quotes are common, no semicolons
- Error handling: explicit `try/except Exception as e` with context; no silent failures
- Script execution: many modules support `if __name__ == "__main__":`

### 4.3 Frontend React/JSX
Observed in `frontend/src/App.jsx`, `frontend/src/components/ChatInterface.jsx`, `frontend/src/main.jsx`.
- ESM project (`"type": "module"`) with functional components + hooks
- Naming: `PascalCase` components, `camelCase` vars/functions, `UPPER_SNAKE_CASE` constants
- Formatting: 2-space indentation, single quotes common, semicolon usage mixed (follow local file)
- Styling: Tailwind utility classes by default; dark mode via root `dark` class
- API: direct `fetch()` + SSE stream parsing are established patterns
- Lint (`frontend/eslint.config.js`):
  - `react-hooks` recommended rules enabled
  - `react-refresh` Vite rules enabled
  - `no-unused-vars` is error (`varsIgnorePattern: '^[A-Z_]'`)

---

## 5) Critical Guardrails

### 5.1 Runtime invariants (backend)
- `setup_env.setup_cuda_dlls()` must run **before** torch/onnxruntime imports
- Preserve this ordering in startup and embedding-related code
- Do not mix Windows and WSL virtualenvs in one runtime flow

### 5.2 Imports / dependencies / paths
- Keep import order and spacing aligned with nearby files
- In backend reusable modules, prefer absolute `backend.*` imports
- Avoid machine-specific absolute paths in committed source
- Check existing dependencies before adding new packages
- Do not add frontend deps if existing stack already covers the use case

### 5.3 Error handling expectations
- No empty `catch` / `except` blocks
- Include enough context for fast debugging
- Backend: keep fallback behavior while surfacing root causes
- Frontend: show safe user-facing fallback text on API/network failures

---

## 6) Verification Checklist
Before edits:
- Confirm target app (`backend/` vs `frontend/`)
- Check nearest subtree AGENTS guide for local constraints
- Confirm command context (repo root vs `frontend/`)
After edits:
- Frontend: run `npm run lint` (and `npm run build` when build-affecting)
- Backend: run affected scripts and relevant `backend/tests/test_*.py` files
- Cross-app changes: verify both sides touched by the change

## 7) Anti-Patterns to Avoid
- Assuming pytest selector workflows (`-k`, markers) without evidence
- Breaking CUDA/ONNX/TensorRT initialization ordering
- Introducing non-Tailwind styling patterns that conflict with existing UI
- Committing secrets (`.env`) or heavy generated/model artifacts

## 8) Quick Pointers
- API/chat streaming: `backend/main.py`
- Vector retrieval manager: `backend/services/vector_store.py`
- CUDA/TensorRT setup: `backend/core/setup_env.py`
- Frontend SSE chat flow: `frontend/src/components/ChatInterface.jsx`
- Citation UI card: `frontend/src/components/ReferenceCard.jsx`

## 9) Maintenance Rule
When build/lint/test scripts, architecture, or runtime assumptions change,
update this file in the same PR so future agents receive accurate guidance.
