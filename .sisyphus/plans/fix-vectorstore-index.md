# Fix VectorStoreManager Missing Index Attribute

## TL;DR

> **Quick Summary**: Add missing `self.index` attribute to `VectorStoreManager.__init__()` so `ingest_pdfs.py` can call `vsm.index.upsert()` without AttributeError.
> 
> **Deliverables**: 
> - One-line fix in `backend/services/vector_store.py` (add `self.index = self.pc.Index(self.index_name)`)
> - Validation that ingest_pdfs.py runs without AttributeError
> 
> **Estimated Effort**: Quick (single-line fix, minimal validation)
> **Parallel Execution**: NO - sequential (only 1 task + validation)
> **Critical Path**: Fix → Validate

---

## Context

### Original Request
User reported error when running `python ingest_pdfs.py`:
```
[Error] Upsert failed for batch 0: 'VectorStoreManager' object has no attribute 'index'
```

### Root Cause Analysis
`ingest_pdfs.py` line 228 calls `vsm.index.upsert(vectors=batch)`, but `VectorStoreManager.__init__()` only creates:
- `self.pc` (Pinecone client)
- `self.vector_store` (LangChain PineconeVectorStore wrapper)

Missing: `self.index` (raw Pinecone Index object for SDK-level operations)

### Impact Scope
- **Affected file**: `backend/services/vector_store.py` (VectorStoreManager class)
- **Only user**: `backend/scripts/ingest_pdfs.py` line 228
- **Grep verification**: No other files instantiate VectorStoreManager or use `.index` attribute

### Architecture Context
Two ingestion paths coexist (both intentional):
1. **Raw Pinecone SDK** (ingest_pdfs.py): Manual embedding → raw vectors → `index.upsert()`
2. **LangChain wrapper** (upsert_chunks): Text strings → LangChain delegates embedding → `vector_store.add_texts()`

Fix adds `self.index` to support path #1 without breaking path #2.

### Metis Review
**Critical Decisions**:
- Eager loading (assign in `__init__`) — only 1 instantiation site, already try/except wrapped
- No extra error handling — let SDK exceptions propagate to existing try/except
- No validation logic — Pinecone SDK provides clear errors if index doesn't exist

---

## Work Objectives

### Core Objective
Expose raw Pinecone Index object via `VectorStoreManager.index` attribute so `ingest_pdfs.py` can perform bulk upsert operations.

### Concrete Deliverables
- `backend/services/vector_store.py` line 46: Add `self.index = self.pc.Index(self.index_name)`

### Definition of Done
- [ ] `python backend/scripts/ingest_pdfs.py --cap 1 --skip-upload` runs without AttributeError
- [ ] `vsm.index` attribute exists and is not None after VectorStoreManager init
- [ ] Existing `vsm.vector_store` (LangChain wrapper) still works

### Must Have
- Exactly one line added at line 46 in vector_store.py
- No changes to any other methods or attributes
- No changes to error handling patterns

### Must NOT Have (Guardrails from Metis)
- ❌ Refactoring of `upsert_chunks()` or other methods
- ❌ Type hints (none exist in surrounding code)
- ❌ Logging statements
- ❌ Comments explaining the fix
- ❌ Validation logic (Pinecone SDK handles this)
- ❌ Retries or error handling
- ❌ Renaming of `self.pc` or `self.vector_store`
- ❌ Changes to LangChain wrapper usage

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (per AGENTS.md: "No formal test suite exists")
- **Automated tests**: None
- **Framework**: N/A

### QA Policy
All acceptance criteria are **agent-executable**. Zero human intervention.

Evidence saved to `.sisyphus/evidence/fix-vectorstore-index/`.

---

## Execution Strategy

### Sequential Execution (1 task + final validation)

```
Task 1: Add self.index attribute
└── No dependencies (standalone fix)

Final Verification: Run all QA scenarios
└── Depends on Task 1
```

---

## TODOs

- [ ] 1. Add `self.index` attribute to VectorStoreManager

  **What to do**:
  - Open `backend/services/vector_store.py`
  - Find line 45: `)`  (closing paren of `pc.create_index()` call)
  - Insert new line 46: `self.index = self.pc.Index(self.index_name)` (blank line before it)
  - Ensure 8-space indent (2 levels: class + method)
  - No trailing whitespace
  - Save file

  **Must NOT do**:
  - Do NOT add type hints
  - Do NOT add comments
  - Do NOT refactor other methods
  - Do NOT change error handling
  - Do NOT rename variables
  - Do NOT add logging

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-line fix in one file, no complex logic
  - **Skills**: None
    - Reason: Straightforward attribute assignment, no specialized domain
  - **Skills Evaluated but Omitted**: N/A (trivial task)

  **Parallelization**:
  - **Can Run In Parallel**: NO (only 1 implementation task)
  - **Parallel Group**: N/A
  - **Blocks**: Final Verification
  - **Blocked By**: None

  **References**:

  **Existing Code Pattern**:
  - `backend/services/vector_store.py:27` - `self.pc = Pinecone(api_key=self.api_key)` (Pinecone client init)
  - `backend/services/vector_store.py:39-45` - Index existence check + creation logic
  - `backend/services/vector_store.py:47-51` - `self.vector_store = PineconeVectorStore(...)` (LangChain wrapper)

  **Usage Pattern**:
  - `backend/scripts/ingest_pdfs.py:228` - `vsm.index.upsert(vectors=batch)` (how .index will be used)

  **Pinecone SDK Version**:
  - `backend/requirements.txt` - `pinecone==7.0.1` (confirms `pc.Index(name)` signature)

  **WHY Each Reference Matters**:
  - Line 27: Shows `self.pc` is already initialized — new attribute uses it
  - Lines 39-45: Index creation/check happens before our line — safe to retrieve Index
  - Lines 47-51: LangChain wrapper comes after — our line goes between pc init and wrapper
  - ingest_pdfs:228: Shows exact usage pattern — `.upsert(vectors=...)` on Index object
  - requirements.txt: Pinecone v7 uses `Index(name)` not `Index.from_name()` — correct signature

  **Acceptance Criteria**:

  **Agent-Executable QA Scenarios (MANDATORY)**:

  ```
  Scenario: VectorStoreManager init creates index attribute
    Tool: Bash (python -c one-liner)
    Preconditions: backend/.env has valid PINECONE_API_KEY and PINECONE_INDEX_NAME
    Steps:
      1. Import setup_env and call setup_cuda_dlls() (MUST be first)
      2. Import VectorStoreManager
      3. Instantiate: vsm = VectorStoreManager()
      4. Assert hasattr(vsm, 'index') is True
      5. Assert vsm.index is not None
    Expected Result: Both assertions pass, print "PASS: index attribute exists"
    Failure Indicators: AssertionError, AttributeError, or any exception
    Evidence: .sisyphus/evidence/fix-vectorstore-index/qa1-index-exists.txt
    Command:
      python -c "
      from backend.core.setup_env import setup_cuda_dlls; setup_cuda_dlls()
      from backend.services.vector_store import VectorStoreManager
      vsm = VectorStoreManager()
      assert hasattr(vsm, 'index'), 'Missing index attribute'
      assert vsm.index is not None, 'Index is None'
      print('PASS: index attribute exists')
      " > .sisyphus/evidence/fix-vectorstore-index/qa1-index-exists.txt 2>&1

  Scenario: ingest_pdfs.py runs without AttributeError
    Tool: Bash (run script with --skip-upload, grep stderr)
    Preconditions: At least one PDF exists in backend/data/pdfs/ OR --cap flag skips missing files gracefully
    Steps:
      1. Run: python backend/scripts/ingest_pdfs.py --cap 1 --skip-upload
      2. Capture stderr
      3. Grep for "AttributeError.*index"
      4. Assert grep exit code is 1 (no match)
    Expected Result: Script runs, no "AttributeError: 'VectorStoreManager' object has no attribute 'index'" in output
    Failure Indicators: Grep finds "AttributeError.*index" string in output
    Evidence: .sisyphus/evidence/fix-vectorstore-index/qa2-no-attr-error.txt
    Command:
      python backend/scripts/ingest_pdfs.py --cap 1 --skip-upload 2>&1 | \
        tee .sisyphus/evidence/fix-vectorstore-index/qa2-no-attr-error.txt | \
        grep -q "AttributeError.*index" && echo "FAIL: AttributeError found" || echo "PASS: No AttributeError"

  Scenario: LangChain wrapper (vector_store) still works
    Tool: Bash (python -c one-liner)
    Preconditions: backend/.env has valid credentials
    Steps:
      1. Import and init VectorStoreManager
      2. Assert hasattr(vsm, 'vector_store') is True
      3. Assert vsm.vector_store is not None
      4. Assert type(vsm.vector_store).__name__ == 'PineconeVectorStore'
    Expected Result: All assertions pass, print "PASS: LangChain wrapper intact"
    Failure Indicators: Any assertion failure or exception
    Evidence: .sisyphus/evidence/fix-vectorstore-index/qa3-wrapper-intact.txt
    Command:
      python -c "
      from backend.core.setup_env import setup_cuda_dlls; setup_cuda_dlls()
      from backend.services.vector_store import VectorStoreManager
      vsm = VectorStoreManager()
      assert hasattr(vsm, 'vector_store'), 'Missing vector_store'
      assert vsm.vector_store is not None, 'vector_store is None'
      assert type(vsm.vector_store).__name__ == 'PineconeVectorStore', f'Wrong type: {type(vsm.vector_store).__name__}'
      print('PASS: LangChain wrapper intact')
      " > .sisyphus/evidence/fix-vectorstore-index/qa3-wrapper-intact.txt 2>&1

  Scenario: No new init exceptions introduced
    Tool: Bash (python -c with try/except)
    Preconditions: backend/.env has valid credentials
    Steps:
      1. Wrap VectorStoreManager init in try/except
      2. If exception: print FAIL and exception message, exit 1
      3. If success: print PASS
    Expected Result: Init succeeds, no exceptions beyond existing Pinecone auth/network errors
    Failure Indicators: New exception types not related to Pinecone connectivity
    Evidence: .sisyphus/evidence/fix-vectorstore-index/qa4-no-new-exceptions.txt
    Command:
      python -c "
      from backend.core.setup_env import setup_cuda_dlls; setup_cuda_dlls()
      try:
          from backend.services.vector_store import VectorStoreManager
          vsm = VectorStoreManager()
          print('PASS: Init successful')
      except Exception as e:
          print(f'FAIL: {e}')
          exit(1)
      " > .sisyphus/evidence/fix-vectorstore-index/qa4-no-new-exceptions.txt 2>&1
  ```

  **Evidence to Capture**:
  - [ ] QA1: `.sisyphus/evidence/fix-vectorstore-index/qa1-index-exists.txt` (stdout: "PASS: index attribute exists")
  - [ ] QA2: `.sisyphus/evidence/fix-vectorstore-index/qa2-no-attr-error.txt` (full ingest_pdfs output, should NOT contain AttributeError)
  - [ ] QA3: `.sisyphus/evidence/fix-vectorstore-index/qa3-wrapper-intact.txt` (stdout: "PASS: LangChain wrapper intact")
  - [ ] QA4: `.sisyphus/evidence/fix-vectorstore-index/qa4-no-new-exceptions.txt` (stdout: "PASS: Init successful")

  **Commit**: YES
  - Message: `fix(vector_store): add missing index attribute for raw Pinecone SDK upsert`
  - Files: `backend/services/vector_store.py`
  - Pre-commit: `python -c "from backend.core.setup_env import setup_cuda_dlls; setup_cuda_dlls(); from backend.services.vector_store import VectorStoreManager; vsm = VectorStoreManager(); assert hasattr(vsm, 'index')"`

---

## Final Verification Wave

> All 4 scenarios run in PARALLEL. ALL must PASS.

- [ ] F1. **Evidence File Verification** — `quick`
  Check that all 4 evidence files exist in `.sisyphus/evidence/fix-vectorstore-index/` and contain "PASS" strings (qa1, qa3, qa4) or no AttributeError (qa2). Read each file, grep for success indicators. If any missing or contains "FAIL", reject with file path + reason.
  Output: `Files [4/4] | QA1 [PASS/FAIL] | QA2 [PASS/FAIL] | QA3 [PASS/FAIL] | QA4 [PASS/FAIL] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Diff Review** — `quick`
  Run `git diff backend/services/vector_store.py`. Verify: (1) Exactly ONE line added, (2) Line matches `self.index = self.pc.Index(self.index_name)`, (3) No other changes (no comments, type hints, refactoring), (4) Correct indentation (8 spaces). Reject if any condition fails.
  Output: `Lines Added [1] | Pattern Match [YES/NO] | Clean Diff [YES/NO] | VERDICT`

- [ ] F3. **Scope Compliance Check** — `quick`
  Verify NO forbidden changes: grep for new logging statements, grep for type hints on new line, check if upsert_chunks or other methods were touched. Read git diff, count modified methods (should be 0 — only `__init__` touched, and only 1 line added). Reject if any forbidden pattern found.
  Output: `Modified Methods [0 expected] | Forbidden Patterns [CLEAN/N found] | VERDICT`

- [ ] F4. **Integration Smoke Test** — `quick`
  Run: `python backend/scripts/ingest_pdfs.py --cap 1 --skip-upload --force-parse 2>&1 | tail -20`. Check last 20 lines for success indicators ("Queued Cap 1", no exceptions). This verifies the full integration: sys.path fix + setup_env + VectorStoreManager + ingest script. Capture exit code.
  Output: `Exit Code [N] | Cap Queued [YES/NO] | Exceptions [CLEAN/N found] | VERDICT`

---

## Commit Strategy

- **1**: `fix(vector_store): add missing index attribute for raw Pinecone SDK upsert` — `backend/services/vector_store.py`, pre-commit: `python -c "from backend.core.setup_env import setup_cuda_dlls; setup_cuda_dlls(); from backend.services.vector_store import VectorStoreManager; vsm = VectorStoreManager(); assert hasattr(vsm, 'index')"`

---

## Success Criteria

### Verification Commands
```bash
# All 4 QA scenarios from Task 1 must PASS (see QA Scenarios section)
# All 4 Final Verification checks must APPROVE
```

### Final Checklist
- [ ] `self.index` attribute exists in VectorStoreManager
- [ ] `ingest_pdfs.py --skip-upload` runs without AttributeError
- [ ] `self.vector_store` (LangChain wrapper) unchanged and functional
- [ ] Exactly one line added, no other changes
- [ ] No logging, type hints, comments, or refactoring
- [ ] All 4 evidence files captured with PASS indicators
