# HK Legal Chatbot Backend

This is the FastAPI backend for the Hong Kong Legal RAG system.

## Setup

### Windows (recommended for daily backend development)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important:
- Keep one runtime per terminal session: Windows PowerShell/CMD with `.venv`.
- Do not reuse a `.venv` created inside WSL for Windows execution.
- If you ever see mixed paths like `/usr/bin\python.exe`, recreate `.venv` from Windows.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If you're on WSL/Linux and using Python 3.12+, this project requires plain
   `torch==2.6.0` wheels (without `+cu124`). The requirements file already
   handles this using platform markers. On Windows, CUDA 12.4 wheels are used.

2. **Recommended venv setup (WSL/Linux)**:
   ```bash
   python3 -m venv .venv-wsl
   source .venv-wsl/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file in this directory and add your DeepSeek API key:
   ```env
   DEEPSEEK_API_KEY=your_api_key_here
   ```

4. **Run the Server**:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000`.

## Run Commands (from repo root)

```powershell
python backend\main.py
python backend\llm_evaluate.py
python backend\scripts\ingest_pdfs.py --cap 282 599A
python backend\tests\test_dll.py
python backend\tests\test_embedding_similarity.py
python backend\tests\test_tensorrt_embedding.py
```

## Windows + WSL split workflow (recommended)

- Use **Windows `.venv`** for daily backend runs (`main.py`, `llm_evaluate.py`, tests).
- Use **WSL-only venv** when you intentionally run Linux/WSL-specific ingestion or TRT experiments.
- Never reuse the same virtualenv across Windows and WSL.

If you accidentally created `.venv` from WSL and see mixed paths like `/usr/bin\\python.exe`, rebuild from Windows:

```powershell
deactivate
Rename-Item .venv .venv_wsl_backup
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

## API Endpoints

- `GET /`: Health check.
- `POST /chat`: Send a message to the chatbot (Streaming).
  - Request Body: `{"message": "your question"}`
  - Response: `text/event-stream` with data chunks: `data: {"answer": "..."}`
