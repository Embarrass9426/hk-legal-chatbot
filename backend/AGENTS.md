# BACKEND KNOWLEDGE BASE

## OVERVIEW
Legal document processing and search backend using FastAPI, ONNX-based LLMs, and vector storage.

## STRUCTURE
```
backend/
├── core/         # Shared utilities (setup_env, embedding_shared, utils)
├── services/     # Business logic (vector_store, embedding_service)
├── parsers/      # PDF parsing (pdf_parser.py — PDFLegalParserV2)
├── scripts/      # CLI tools (ingest_pdfs, batch_download, cap_discovery)
├── tests/        # Ad-hoc test scripts (NOT pytest — __main__ guards)
├── bin/          # Bundled Poppler binaries for PDF to Image conversion
├── data/         # PDF sources (pdfs/) and parsed text/JSON (parsed/)
├── models/       # Local embedding models (Yuan ONNX/TRT)
└── main.py       # FastAPI app entry point
```

## WHERE TO LOOK
| Component | File | Description |
|-----------|------|-------------|
| API Layer | `main.py` | FastAPI endpoints (`/chat` POST SSE, `/` GET health) |
| Batch Ingestion | `scripts/ingest_pdfs.py` | CLI: parse → embed → upsert to Pinecone |
| Vector DB | `services/vector_store.py` | `VectorStoreManager` — Pinecone + embeddings |
| Embedding | `services/embedding_service.py` | Singleton ONNX/TRT service, mean pooling |
| PDF Parsing | `parsers/pdf_parser.py` | `PDFLegalParserV2` — unstructured-based |
| Query Rewrite | `core/utils.py` | LLM-powered query rewrite for retrieval |
| CUDA Setup | `core/setup_env.py` | Must run before torch/onnx imports |
| Shared Logic | `core/embedding_shared.py` | Shared embedding queues/utilities |

## CONVENTIONS
- **Execution**: Most modules can be run directly as scripts for testing (check `__main__` guards).
- **Tooling**: Use `uv` for package management. Requirements are in `requirements.txt` (compiled by `uv pip compile`).
- **Environment**: `setup_env.setup_cuda_dlls()` MUST be called before importing torch/onnxruntime.
- **Imports**: Prefer `backend.*` absolute imports. `main.py` uses bare imports (`import utils`) due to `sys.path` manipulation — don't follow this pattern in new code.

## ANTI-PATTERNS
- Avoid absolute paths in scripts; use `os.path` relative to `__file__`.
- Do not commit large PDF files, model binaries, or `.env` to git.
- Do not import torch/onnxruntime before calling `setup_env.setup_cuda_dlls()`.

## NOTES
- Poppler binaries in `bin/` are required for PDF parsing.
- Models in `models/` are ONNX format with optional TensorRT acceleration.
