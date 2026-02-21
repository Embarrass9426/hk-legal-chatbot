# Fix Torch CUDA Device Placement Causing Zero-Vector Embeddings

## TL;DR

> **Quick Summary**: Embedding service fails when PyTorch is CPU-only but tries to move tokenizer tensors to CUDA device. This causes all-zero embeddings and triggers our safety abort. Remove explicit `.to(device)` calls since ONNX Runtime handles device placement internally.
> 
> **Deliverables**:
> - Fixed `backend/services/embedding_service.py` (remove lines 148-156)
> - Verification test confirming non-zero embeddings
> 
> **Estimated Effort**: Quick (5-10 minutes)
> **Parallel Execution**: NO - single file, sequential
> **Critical Path**: Fix → Test → Verify

---

## Context

### Original Request
User reported: "Error processing Cap 260B: Embedding failed: Embedding model produced all-zero vectors for entire batch (2 texts). Model may be broken — aborting."

### Root Cause Analysis
**Discovered Issue**: Lines 154-156 in `embedding_service.py` explicitly move tokenizer output tensors to `self.model.device`:
```python
device = self.model.device
for k, v in inputs.items():
    inputs[k] = v.to(device)
```

**Why It Breaks**:
1. ONNX Runtime with TensorRT provider works fine (loads model, runs inference)
2. PyTorch installation is **CPU-only** (`torch.cuda.is_available() == False`)
3. When code calls `.to(device)` where `device="cuda"`, PyTorch raises: `AssertionError: Torch not compiled with CUDA enabled`
4. This exception is caught somewhere upstream, causing silent failure → zero embeddings
5. Our all-zero guard (lines 181-185) correctly detects and aborts

**Why Explicit Device Move Is Unnecessary**:
- ONNX Runtime (`ORTModelForFeatureExtraction`) handles device placement internally
- TensorRT execution provider manages GPU memory without PyTorch CUDA
- Tokenizer outputs are CPU tensors by default, which ONNX Runtime accepts
- The explicit `.to(device)` was cargo-culted from pure PyTorch code

### Test Evidence
Ran direct test:
```bash
python -c "service.embed_documents(['test'])"
```
Result: `AssertionError: Torch not compiled with CUDA enabled` at line 156

---

## Work Objectives

### Core Objective
Remove explicit device placement logic that assumes PyTorch CUDA is available, allowing ONNX Runtime to handle device management internally.

### Concrete Deliverables
- **File Modified**: `backend/services/embedding_service.py` (lines 148-156 removed, replaced with 2-line comment)
- **Verification Test**: Script that generates embeddings for Cap 260B chunks and confirms non-zero vectors

### Definition of Done
- [x] Lines 148-156 removed from `embedding_service.py`
- [x] Comment added explaining why device move is unnecessary
- [x] Test script confirms embeddings are non-zero for Cap 260B chunks
- [x] `python backend/scripts/ingest_pdfs.py --cap 260B` runs without zero-vector abort

### Must Have
- Remove `.to(device)` calls (lines 154-156)
- Keep all other logic unchanged (tokenization, inference, pooling, normalization, zero guards)

### Must NOT Have (Guardrails)
- No changes to tokenization logic
- No changes to pooling/normalization logic
- No changes to zero-vector guards (lines 179-187)
- No changes to model loading logic
- No removal of `self.device` attribute (may be used elsewhere)
- No addition of try/except around device placement (fix root cause, don't mask it)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO (no formal test suite)
- **Automated tests**: None (ad-hoc verification scripts)
- **Framework**: N/A
- **If TDD**: N/A

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/fix-torch-cuda/`.

- **Module Test**: Use Bash (python REPL) — Import service, embed test texts, verify non-zero
- **Integration Test**: Use Bash (run ingestion script) — Process Cap 260B, check for zero-vector abort

---

## Execution Strategy

### Sequential Execution (No Parallelism)

Single file change with immediate verification.

```
Task 1: Remove device placement logic
  ↓
Task 2: Verify embeddings are non-zero
  ↓
Task 3: Test full ingestion pipeline
```

---

## TODOs

- [x] 1. Remove explicit device placement from embedding_service.py

  **What to do**:
  - Open `backend/services/embedding_service.py`
  - Locate lines 148-156 (comment block + device assignment + for loop)
  - Replace with 2-line comment:
    ```python
    # ONNX Runtime handles device placement internally - no need to move tensors
    # Explicit .to(device) fails if torch is CPU-only while ORT uses TensorRT
    ```
  - Save file

  **Must NOT do**:
  - Do not modify any other lines
  - Do not remove `self.device = "cuda" if torch.cuda.is_available() else "cpu"` at line 31 (may be used elsewhere)
  - Do not add try/except around this section
  - Do not change inference call at line 160
  - Do not change pooling/normalization logic (lines 165-177)
  - Do not touch zero-vector guards (lines 179-187)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file edit, remove 9 lines, add 2 lines comment
  - **Skills**: None needed
  - **Skills Evaluated but Omitted**: None

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2 (verification depends on this fix)
  - **Blocked By**: None (can start immediately)

  **References**:

  **File to Modify**:
  - `backend/services/embedding_service.py:148-156` — Lines to remove/replace

  **Context for Understanding**:
  - `backend/services/embedding_service.py:31` — Where `self.device` is set (do NOT modify)
  - `backend/services/embedding_service.py:160` — Inference call (comes after the device move)
  - `backend/services/embedding_service.py:179-187` — Zero-vector guards (keep unchanged)

  **Why This Fix Works**:
  - ONNX Runtime documentation: "ORTModel handles device placement automatically when using TensorRT/CUDA providers"
  - Optimum library source code: `ORTModelForFeatureExtraction.__call__()` accepts CPU tensors and moves internally
  - The explicit `.to(device)` was unnecessary defensive programming that assumes PyTorch CUDA

  **Acceptance Criteria**:

  **Syntax Validation**:
  - [x] `python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"`
  - [x] Output: no errors (silent success)

  **QA Scenarios**:

  ```
  Scenario: Embedding service loads and generates non-zero vectors for test input
    Tool: Bash (python inline script)
    Preconditions: backend/services/embedding_service.py has device move logic removed
    Steps:
      1. Run python script:
         ```
         python -c "
         import sys; sys.path.insert(0, '.');
         from backend.core import setup_env; setup_env.setup_cuda_dlls();
         from backend.services.embedding_service import get_embedding_service;
         service = get_embedding_service();
         vectors = service.embed_documents(['Hong Kong legal document test']);
         import numpy as np;
         arr = np.array(vectors[0]);
         is_zero = np.all(arr == 0);
         norm = np.linalg.norm(arr);
         print(f'Zero: {is_zero}, Norm: {norm:.6f}');
         assert not is_zero, 'Vector is all zeros!';
         assert norm > 0.5, f'Norm too low: {norm}';
         print('PASS')
         " 2>&1 | tail -1
         ```
      2. Expect last line output: `PASS`
    Expected Result: Vector is non-zero with norm > 0.5
    Failure Indicators: `AssertionError`, `Zero: True`, norm < 0.1
    Evidence: .sisyphus/evidence/fix-torch-cuda/task-1-embedding-test.txt (save stdout)

  Scenario: Cap 260B chunks produce non-zero embeddings (real data test)
    Tool: Bash (python inline script)
    Preconditions: backend/data/parsed/cap260B.json exists with 2 chunks
    Steps:
      1. Run python script that loads actual Cap 260B chunks and embeds them:
         ```
         python -c "
         import sys, json; sys.path.insert(0, '.');
         from backend.core import setup_env; setup_env.setup_cuda_dlls();
         from backend.services.embedding_service import get_embedding_service;
         service = get_embedding_service();
         chunks = json.load(open('backend/data/parsed/cap260B.json', encoding='utf-8'));
         texts = [c['content'] for c in chunks];
         print(f'Testing {len(texts)} chunks...');
         vectors = service.embed_documents(texts);
         import numpy as np;
         for i, v in enumerate(vectors):
             arr = np.array(v);
             is_zero = np.all(arr == 0);
             norm = np.linalg.norm(arr);
             print(f'Chunk {i}: Zero={is_zero}, Norm={norm:.6f}');
             assert not is_zero, f'Chunk {i} is all zeros!';
         print('PASS')
         " 2>&1 | grep -E '(Testing|Chunk|PASS)'
         ```
      2. Expect output showing 2 chunks, both with Zero=False and norm > 0.5
      3. Last line: `PASS`
    Expected Result: All chunks produce non-zero embeddings
    Failure Indicators: `Zero=True`, `AssertionError`, norm < 0.1
    Evidence: .sisyphus/evidence/fix-torch-cuda/task-1-cap260b-test.txt
  ```

  **Evidence to Capture**:
  - [x] .sisyphus/evidence/fix-torch-cuda/task-1-embedding-test.txt (scenario 1 output)
  - [x] .sisyphus/evidence/fix-torch-cuda/task-1-cap260b-test.txt (scenario 2 output)

  **Commit**: YES
  - Message: `fix(embedding): remove torch device placement breaking CPU-only torch with TensorRT`
  - Files: `backend/services/embedding_service.py`
  - Pre-commit: `python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"`

---

## Final Verification Wave

> After implementation task completes, run these 2 parallel reviews. Both must APPROVE.

- [x] F1. **Code Quality Review** — `unspecified-high`
  
  Read `backend/services/embedding_service.py` lines 148-160. Verify:
  1. Lines 148-156 (old device move logic) are gone
  2. New 2-line comment explains why device move is removed
  3. Line 160 inference call unchanged: `outputs = self.model(**inputs)`
  4. No try/except added around inference
  5. Lines 165-187 (pooling + normalization + zero guards) unchanged
  
  Run syntax validation:
  ```bash
  python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"
  ```
  
  Output: `Code Quality: APPROVE | REJECT (reason)`

- [x] F2. **Integration Test** — `unspecified-high`
  
  Run full ingestion for Cap 260B:
  ```bash
  cd backend/scripts && python ingest_pdfs.py --cap 260B
  ```
  
  Expected:
  - No `AssertionError: Torch not compiled with CUDA enabled`
  - No `Embedding model produced all-zero vectors for entire batch`
  - Success: `Upserted batch 0 for Cap 260B (2 vectors)`
  
  Output: `Integration Test: PASS | FAIL (error message)`

---

## Commit Strategy

Single atomic commit after Task 1 completes and QA scenarios pass.

- **Message**: `fix(embedding): remove torch device placement breaking CPU-only torch with TensorRT`
- **Files**: `backend/services/embedding_service.py`
- **Pre-commit**: Syntax validation via `ast.parse()`

---

## Success Criteria

### Verification Commands
```bash
# 1. Syntax valid
python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"
# Expected: silent success

# 2. Embeddings non-zero
python -c "import sys; sys.path.insert(0, '.'); from backend.core import setup_env; setup_env.setup_cuda_dlls(); from backend.services.embedding_service import get_embedding_service; service = get_embedding_service(); v = service.embed_documents(['test']); import numpy as np; assert np.linalg.norm(v[0]) > 0.5; print('PASS')"
# Expected: PASS

# 3. Ingestion succeeds
cd backend/scripts && python ingest_pdfs.py --cap 260B
# Expected: No zero-vector abort, successful upsert
```

### Final Checklist
- [x] Lines 148-156 removed from embedding_service.py
- [x] Comment added explaining why device move is unnecessary
- [x] Syntax validation passes
- [x] Test embeddings are non-zero (norm > 0.5)
- [x] Cap 260B ingestion succeeds without zero-vector abort
- [x] Code committed with descriptive message
