# Fix Zero-Vector Rejection by Pinecone

## TL;DR

> **Quick Summary**: Pinecone is rejecting vectors with all zeros (e.g., `cap3_chunk_2`). The embedding service needs a guard to detect and fix zero vectors before normalization, preventing Pinecone 400 errors during upsert.
> 
> **Deliverables**:
> - Zero-vector detection and fix in `backend/services/embedding_service.py`
> - Successful ingestion of Cap 3 without 400 errors
> 
> **Estimated Effort**: Quick (single function edit, 5 lines)
> **Parallel Execution**: NO - single file edit
> **Critical Path**: Add guard → Test ingestion

---

## Context

### Original Request
User encountered Pinecone 400 error:
```
Upsert failed for batch 0: (400) Bad Request
Dense vectors must contain at least one non-zero value. 
Vector ID 'cap3_chunk_2' contains only zeros
```

### Investigation Summary
- **Root Cause**: `embedding_service.py` normalizes embeddings with `F.normalize()`, which preserves zero vectors (doesn't error, returns zeros).
- **Missing Guard**: Old `ingest_legal_pdfs.py` had explicit zero-detection and random noise injection (`emb[zero_rows] += torch.rand_like(...) * 1e-6`).
- **Current State**: New refactored `embedding_service.py` (singleton pattern) is missing this guard.

### Evidence
- Cap 3, chunk 2 has valid content (200 chars verified via Python check).
- Embedding service reached normalization step but produced zero vector.
- Pinecone rejects zero vectors with 400 error.

---

## Work Objectives

### Core Objective
Add zero-vector guard to `embedding_service.py` to inject minimal random noise into any all-zero embedding before returning to caller.

### Concrete Deliverables
- Modified `backend/services/embedding_service.py` with zero-vector detection guard
- Verified ingestion of Cap 3 completes without Pinecone 400 errors

### Definition of Done
- [ ] Guard added after normalization step in `_embed_batch` method
- [ ] `python backend/scripts/ingest_pdfs.py --cap 3 --force-parse` completes successfully
- [ ] No "contains only zeros" errors in output

### Must Have
- Zero-detection: `embeddings.abs().sum(dim=1) == 0`
- Noise injection: `embeddings[zero_rows] += torch.rand_like(embeddings[zero_rows]) * 1e-6`
- NaN cleanup: `torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)`

### Must NOT Have (Guardrails)
- No changes to normalization logic (keep `F.normalize`)
- No changes to pooling logic (keep mean pooling)
- No changes to tokenization (keep 512 max_length)
- No logging additions
- No type hints additions
- No comments explaining the guard

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (no test framework for backend)
- **Automated tests**: None
- **Framework**: N/A

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/fix-zero-vector/`.

- **Backend/Script**: Use Bash — Run ingestion command, parse output, assert no errors

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Single Task):
└── Task 1: Add zero-vector guard to embedding_service._embed_batch [quick]

Wave FINAL (After Task 1):
├── Task F1: Syntax validation [quick]
├── Task F2: Import chain test [quick]
├── Task F3: Cap 3 ingestion smoke test [unspecified-high]
└── Task F4: Verify no zero-vector errors in logs [quick]

Critical Path: Task 1 → F1 → F2 → F3 → F4
Parallel Speedup: N/A (sequential)
Max Concurrent: 1
```

### Dependency Matrix

- **1**: — — F1, F2, F3, F4
- **F1**: 1 — —
- **F2**: 1 — F3
- **F3**: F2 — F4
- **F4**: F3 — —

---

## TODOs

- [ ] 1. Add zero-vector guard to `embedding_service._embed_batch`

  **What to do**:
  - Open `backend/services/embedding_service.py`
  - Locate line 177: `embeddings = F.normalize(embeddings, p=2, dim=1)`
  - After normalization, BEFORE `return embeddings.cpu().numpy().tolist()`:
    - Add: `embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)`
    - Add: `zero_rows = embeddings.abs().sum(dim=1) == 0`
    - Add: `if zero_rows.any():`
    - Add: `    embeddings[zero_rows] += torch.rand_like(embeddings[zero_rows]) * 1e-6`

  **Must NOT do**:
  - Change normalization method
  - Add print statements or logging
  - Add comments
  - Change pooling logic
  - Modify tokenization parameters

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function edit, 4 lines of code, straightforward logic
  - **Skills**: []
    - Reason: No specialized skills needed for basic tensor operations

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `backend/services/embedding_service.py:177` - Normalization step location (insert guard AFTER this line)
  - Old `backend/ingest_legal_pdfs.py` (deleted, see git diff) - Original zero-guard logic from lines ~90-96 in the deleted file

  **API/Type References**:
  - `torch.nan_to_num` - Replace NaN/inf with specified values
  - `torch.abs().sum(dim=1)` - Row-wise sum of absolute values
  - `torch.rand_like` - Generate random tensor with same shape

  **WHY Each Reference Matters**:
  - Line 177 context: Need to insert guard between normalization and return statement
  - Old implementation: Exact same logic worked in production, just need to port to new location

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Happy path — embedding service handles valid text
    Tool: Bash
    Preconditions: Clean backend state, Cap 3 PDF exists in backend/data/pdfs/
    Steps:
      1. cd C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot
      2. python backend/scripts/ingest_pdfs.py --cap 3 --force-parse
      3. Parse output for "Uploaded Cap 3 vectors to Pinecone" success message
      4. Assert no "contains only zeros" error messages
    Expected Result: Exit code 0, success message present, no zero-vector errors
    Failure Indicators: Exit code != 0, "contains only zeros" in output, Python traceback
    Evidence: .sisyphus/evidence/fix-zero-vector/cap3-ingestion-success.txt

  Scenario: Edge case — empty string input to embedding service
    Tool: Bash (Python REPL)
    Preconditions: Modified embedding_service.py with guard
    Steps:
      1. python -c "import sys; sys.path.append('C:\\Users\\Embarrass\\Desktop\\vscode\\hk-legal-chatbot'); from backend.core import setup_env; setup_env.setup_cuda_dlls(); from backend.services.embedding_service import get_embedding_service; svc = get_embedding_service(); result = svc.embed_documents(['']); import torch; vec = torch.tensor(result[0]); assert vec.abs().sum() > 0, 'Zero vector returned!'; print('PASS: Non-zero vector for empty string')"
    Expected Result: PASS message, no assertion error
    Evidence: .sisyphus/evidence/fix-zero-vector/empty-string-test.txt
  ```

  **Evidence to Capture**:
  - [ ] Cap 3 ingestion terminal output saved to task-1-cap3-ingestion-success.txt
  - [ ] Empty string test output saved to task-1-empty-string-test.txt

  **Commit**: YES
  - Message: `fix(embeddings): add zero-vector guard to prevent Pinecone rejection`
  - Files: `backend/services/embedding_service.py`
  - Pre-commit: `python -c "import ast; ast.parse(open('backend/services/embedding_service.py').read())"`

---

## Final Verification Wave

- [ ] F1. **Syntax Validation** — `quick`
  Run `python -c "import ast; ast.parse(open(r'backend\services\embedding_service.py').read())"` to verify no syntax errors. Exit code must be 0.
  Output: `PASS/FAIL | VERDICT: APPROVE/REJECT`

- [ ] F2. **Import Chain Test** — `quick`
  Run `python -c "import sys; sys.path.append('C:\\Users\\Embarrass\\Desktop\\vscode\\hk-legal-chatbot'); from backend.core import setup_env; setup_env.setup_cuda_dlls(); from backend.services.embedding_service import get_embedding_service; print('Import chain: PASS')"`. Verify "Import chain: PASS" appears.
  Output: `Import [PASS/FAIL] | VERDICT`

- [ ] F3. **Cap 3 Ingestion Smoke Test** — `unspecified-high`
  Run `python backend/scripts/ingest_pdfs.py --cap 3 --force-parse` from project root. Parse output for success message "Uploaded Cap 3 vectors to Pinecone". Assert no "contains only zeros" errors. Capture full output to `.sisyphus/evidence/fix-zero-vector/final-cap3-ingestion.txt`.
  Output: `Ingestion [PASS/FAIL] | Errors [0/N] | VERDICT`

- [ ] F4. **Verify No Zero-Vector Errors** — `quick`
  Grep the ingestion output for "contains only zeros". Count must be 0.
  Run: `python -c "output = open('.sisyphus/evidence/fix-zero-vector/final-cap3-ingestion.txt').read(); assert 'contains only zeros' not in output; print('PASS: No zero-vector errors')"`
  Output: `Zero-vector errors [0/N] | VERDICT`

---

## Commit Strategy

- **1**: `fix(embeddings): add zero-vector guard to prevent Pinecone rejection` — `backend/services/embedding_service.py`, `python -c "import ast; ast.parse(open('backend/services/embedding_service.py').read())"`

---

## Success Criteria

### Verification Commands
```bash
# Syntax check
python -c "import ast; ast.parse(open(r'backend\services\embedding_service.py').read())"  # Expected: no output (exit 0)

# Import check
python -c "import sys; sys.path.append('C:\\Users\\Embarrass\\Desktop\\vscode\\hk-legal-chatbot'); from backend.services.embedding_service import get_embedding_service; print('OK')"  # Expected: OK

# Integration test
python backend/scripts/ingest_pdfs.py --cap 3 --force-parse  # Expected: "Uploaded Cap 3 vectors to Pinecone" with no "contains only zeros" errors
```

### Final Checklist
- [ ] Zero-vector guard added after normalization
- [ ] Cap 3 ingests successfully
- [ ] No Pinecone 400 errors in output
