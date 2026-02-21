# Plan: Yuan Embedding ONNX + TensorRT Migration to WSL

## TL;DR

> **Quick Summary**: Export the Yuan-embedding-2.0-en model to an optimized ONNX format (FP16) and configure the legal chatbot backend to run in WSL2 using `TensorrtExecutionProvider` for maximum inference speed on NVIDIA RTX 5060 Ti.
> 
> **Deliverables**:
> - Updated `setup_env.py` (WSL/Linux support)
> - Optimized `export_yuan_cuda.py` (FP16 + Opset 17 + Dynamic Axes)
> - Updated `vector_store.py` (TensorRT provider options + path normalization)
> - Updated `ingest_legal_pdfs.py` (Aligned with TensorRT loading)
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - Phase 1 and Phase 2 (scripts) can be prepared in parallel.
> **Critical Path**: Export Model → Update Loader → Verify TRT

---

## Context

### Original Request
The user wants to export the Yuan embedding model to ONNX without bugs and run it in WSL using `TensorrtExecutionProvider` on their NVIDIA GPU.

### Interview Summary
**Key Discussions**:
- Environment: WSL2 (Ubuntu) on Windows 11 host.
- GPU: NVIDIA GeForce RTX 5060 Ti.
- Current State: Using `optimum` for export/inference but encountering bugs and lacking TensorRT optimization.
- Hardcoded Paths: The codebase contains absolute Windows paths (e.g., `C:\Users\...`) which break in WSL.

### Research Findings
- **TensorRT on WSL**: Requires `onnxruntime-gpu`, `tensorrt` libs, and proper `LD_LIBRARY_PATH` inside WSL.
- **Provider Names**: ONNX Runtime uses `TensorrtExecutionProvider` (case varies by version, usually `TensorrtExecutionProvider` in Python API options).
- **Optimization**: `trt_fp16_enable=True` and `trt_engine_cache_enable=True` are critical for performance and startup speed.

---

## Work Objectives

### Core Objective
Enable high-performance, bug-free embedding inference using TensorRT on WSL2 for the HK Legal Chatbot.

### Concrete Deliverables
- `backend/setup_env.py`: Cross-platform DLL/library path handler (skips Windows logic on Linux).
- `backend/export_yuan_cuda.py`: Script optimized for TensorRT export (FP16, Opset 17, Dynamic Shapes).
- `backend/vector_store.py`: Updated `BoostedYuanEmbeddings` to use `TensorrtExecutionProvider` with caching.
- `backend/ingest_legal_pdfs.py`: Updated to handle relative model paths.
- Exported FP16 ONNX model in `backend/models/yuan-onnx-trt/`.

### Definition of Done
- [ ] `nvidia-smi` confirms GPU visibility in WSL.
- [ ] `export_yuan_cuda.py` completes without errors in WSL.
- [ ] `vector_store.py` loads the model with `TensorrtExecutionProvider`.
- [ ] Embeddings generated in WSL match quality of PyTorch original.
- [ ] TensorRT engine cache files created in `backend/models/yuan-onnx-trt/cache/`.

### Must Have
- FP16 quantization for the ONNX model.
- Dynamic shapes support (batch size, sequence length).
- Path normalization (removing `C:\` hardcoding).
- Engine caching enabled ( `trt_engine_cache_enable=True`).

### Must NOT Have (Guardrails)
- NO hardcoded absolute paths to Windows user directories.
- NO silent fallback to CPU (must log warning if TensorRT fails).

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL verification is executed by the agent using tools.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None (Script-based verification)
- **Framework**: Python/Bash

### Agent-Executed QA Scenarios (MANDATORY)

Scenario: Verify ONNX Export in WSL
  Tool: Bash
  Preconditions: WSL environment ready, `uv` installed.
  Steps:
    1. Run: `curl -LsSf https://astral.sh/uv/install.sh | sh` (install uv)
    2. Run: `uv venv .wsl_venv`
    3. Run: `uv pip install onnx onnxconverter-common "optimum[onnxruntime-gpu]" transformers`
    4. Run: `uv run backend/export_yuan_trt.py`
    5. Assert: File `backend/models/yuan-onnx-trt/model.onnx` exists.
    6. Assert: Model size is approx 1.2GB (FP16) vs 2.4GB (FP32).
  Expected Result: Export successful.
  Evidence: Terminal output and `ls -lh`.

Scenario: Verify TensorRT Initialization
  Tool: Bash
  Preconditions: Model exported.
  Steps:
    1. Run: `python3 -c "from backend.vector_store import BoostedYuanEmbeddings; b = BoostedYuanEmbeddings(); print(b.session.get_providers())"`
    2. Assert: `TensorrtExecutionProvider` is in the list.
    3. Assert: `backend/models/yuan-onnx-trt/cache/` contains `.engine` or `.cache` files after first run.
  Expected Result: TensorRT active and caching.
  Evidence: Print output and cache directory check.

Scenario: Verify Embedding Dimensions
  Tool: Bash
  Preconditions: `BoostedYuanEmbeddings` initialized.
  Steps:
    1. Run: `python3 -c "from backend.vector_store import BoostedYuanEmbeddings; b = BoostedYuanEmbeddings(); print(len(b.embed_query('test')))"`
    2. Assert: Output is `1024`.
  Expected Result: 1024-dimensional vector returned.
  Evidence: Captured output.

---

## Execution Strategy

### Parallel Execution Waves
Wave 1:
- Task 1: Environment Adaption (setup_env.py)
- Task 2: Path Normalization (vector_store.py, ingest_legal_pdfs.py)

Wave 2:
- Task 3: Optimized Export Script (export_yuan_cuda.py)

Wave 3:
- Task 4: Run Export in WSL

Wave 4:
- Task 5: Integration & Verification

---

## TODOs

- [ ] 1. Update `backend/setup_env.py` for WSL/Linux
  - **What to do**: Add logic to skip Windows DLL loading when on Linux. Add `LD_LIBRARY_PATH` checks if necessary.
  - **Recommended Agent Profile**: `quick` + `python-programmer`
  - **Acceptance Criteria**: `python3 backend/setup_env.py` runs in WSL without WinError.

- [ ] 2. Normalize Paths in `vector_store.py` and `ingest_legal_pdfs.py`
  - **What to do**: Replace `C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\...` with relative paths starting from the project root or use `os.path.join(os.path.dirname(__file__), "models", ...)`.
  - **Recommended Agent Profile**: `quick`
  - **Acceptance Criteria**: No absolute Windows paths remain in these files.

- [ ] 3. Refactor `backend/export_yuan_cuda.py` for TensorRT
  - **What to do**:
    - Update to Opset 17.
    - Enable FP16 export.
    - Define dynamic axes for `input_ids`, `attention_mask`, `position_ids`.
    - Explicitly set `trust_remote_code=True`.
  - **Recommended Agent Profile**: `ultrabrain` + `onnx-specialist`
  - **Acceptance Criteria**: Script is ready for execution in WSL.

- [ ] 4. Execute Export in WSL
  - **What to do**: Run the export script inside the WSL environment.
  - **Recommended Agent Profile**: `quick`
  - **Acceptance Criteria**: `backend/models/yuan-onnx-trt/model.onnx` created.

- [ ] 5. Implement TensorRT Loading Logic
  - **What to do**:
    - Update `BoostedYuanEmbeddings` to use `TensorrtExecutionProvider`.
    - Configure `trt_engine_cache_enable=True`, `trt_engine_cache_path`, and `trt_fp16_enable=True`.
    - Add fallback logic with warnings.
  - **Recommended Agent Profile**: `ultrabrain`
  - **Acceptance Criteria**: `vector_store.py` initialization logs "Using TensorRTExecutionProvider".

- [ ] 6. Final Verification & Benchmarking
  - **What to do**: Run the ingestion pipeline or a chat query in WSL to confirm everything works end-to-end.
  - **Recommended Agent Profile**: `quick`
  - **Acceptance Criteria**: App works in WSL, GPU utilization visible in `nvidia-smi`.

---

## Success Criteria

### Verification Commands
```bash
wsl -e python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
wsl -e python3 backend/main.py
```

### Final Checklist
- [x] Model exported in FP16
- [x] TensorRT active in WSL
- [x] Paths normalized
- [x] Performance improved vs CPU/Standard CUDA
