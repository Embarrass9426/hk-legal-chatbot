# Draft: Backend Reorganization and Ingestion Stabilisation

## Requirements (confirmed)
- Modular structure for `backend/` (already partially done).
- Clean up `requirements.txt` and `.gitignore` (already done).
- Stable `ingest_pdfs.py` execution.
- **Support Direct Execution**: `python ingest_pdfs.py` from within `backend/scripts/` must work.
- **TensorRT in WSL**: Ensure `TensorrtExecutionProvider` works in the WSL environment.
- **Performance Priority**: Embedding generation priority is TensorRT -> CUDA -> CPU.
- **Consolidate Logic**: Move embedding and model loading to `backend/services/embedding_service.py`.
- Update Pinecone vector database with legal PDFs.

## Technical Decisions
- **Directory Structure**: `backend/` organized into `parsers/`, `services/`, `core/`, and `scripts/`.
- **Vector DB**: Pinecone.
- **Embeddings**: Local ONNX models (`yuan-onnx-trt`).
- **Provider Priority**: TensorRT -> CUDA -> CPU.
- **Embedding Service**: A new `EmbeddingService` in `backend/services/embedding_service.py` will handle model lifecycle.
- **Path Logic**: Use `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))` to support direct execution from `scripts/`.

## Research Findings
- `ingest_pdfs.py` has flawed `sys.path` logic (appends `backend/` instead of parent folder).
- `TensorrtExecutionProvider` is failing to initialize in some environments (WSL specific).
- Duplicate model loading logic exists in `ingest_pdfs.py` and `vector_store.py`.

## Open Questions
- None (User confirmed preferences).

## Scope Boundaries
- **INCLUDE**: Fix path logic, optimize GPU providers (TensorRT focus in WSL), consolidate embedding logic into `embedding_service.py`, verify full ingestion run.
- **EXCLUDE**: UI changes, new API endpoints (unless needed for ingestion), architectural changes beyond the backend structure.
