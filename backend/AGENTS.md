# BACKEND KNOWLEDGE BASE

## OVERVIEW
Legal document processing and search backend using FastAPI, ONNX-based LLMs, and vector storage.

## STRUCTURE
```
backend/
├── bin/          # Bundled Poppler binaries for PDF to Image conversion
├── data/         # PDF sources (pdfs/) and parsed text/JSON (parsed/)
├── models/       # Local LLM and embedding models (Yuan ONNX)
├── scripts/      # Environment setup and utility scripts
└── tests/        # Python tests for core modules
```

## WHERE TO LOOK
| Component | File | Description |
|-----------|------|-------------|
| Batch Ingestion | `ingest_legal_pdfs.py` | Main script for processing new PDFs |
| Vector DB | `vector_store.py` | Indexing and querying logic |
| PDF Parsing | `pdf_parser_v2.py` | Unstructured-based parsing logic |
| API Layer | `main.py` | FastAPI endpoints for frontend integration |
| Shared Logic | `embedding_shared.py` | Shared embedding model utilities |

## CONVENTIONS
- **Execution**: Most modules can be run directly as scripts for testing (check `__main__` guards).
- **Tooling**: Use `uv` for package management. Requirements are in `requirements.txt`.
- **Environment**: `setup_env.py` manages local paths and binary availability.

## ANTI-PATTERNS
- Avoid absolute paths in scripts; use `os.path` relative to `backend/`.
- Do not commit large PDF files or model binaries to git.

## NOTES
- Poppler binaries in `bin/` are required for `pdf_parser_v2.py` to function.
- Models in `models/` are typically ONNX format for CUDA/CPU compatibility.
