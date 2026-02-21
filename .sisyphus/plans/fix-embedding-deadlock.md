# Fix Resource Deadlock in EmbeddingService

## TL;DR

> **Quick Summary**: EmbeddingService deadlocks because same lock is used for singleton initialization AND embedding operations. Separate into two locks.
> 
> **Deliverables**:
> - Fixed `backend/services/embedding_service.py` (4 lines changed)
> - Ingestion runs without deadlock
> 
> **Estimated Effort**: 5 minutes (4-line change)
> **Parallel Execution**: NO - single file
> **Critical Path**: Fix → Test

---

## Context

### Error
```
Error processing Cap 1: Embedding failed: resource deadlock would occur
RuntimeError: Embedding failed: resource deadlock would occur
```

### Root Cause
`EmbeddingService` uses **same lock** (`_lock` at line 14) for:
1. **Singleton initialization** in `__new__` (line 18)
2. **Thread-safe embedding** in `embed_documents` (line 127)

**Deadlock scenario**:
- Thread A holds `self._lock` (in `embed_documents`)
- Something triggers singleton access again  
- `__new__` tries to acquire `cls._lock` (same lock!)
- Thread A waits for lock it already holds → **DEADLOCK**

**Why same lock**: `_lock` is class variable. `self._lock` and `cls._lock` reference the same object.

---

## Work Objectives

### Core Objective
Separate singleton initialization lock from embedding operation lock.

### Concrete Deliverables
- Modified `backend/services/embedding_service.py` with two locks
- Cap 1 ingestion completes without deadlock error

### Definition of Done
- [x] `_init_lock` for singleton (class-level)
- [x] `_embed_lock` for embedding ops (instance-level)
- [x] `python backend/scripts/ingest_pdfs.py --cap 1` completes successfully

### Must Have
- Two separate locks (no shared lock)
- Singleton still thread-safe
- Embedding operations still thread-safe

### Must NOT Have
- Don't remove locking (still need thread safety)
- Don't change singleton pattern logic
- Don't change embedding logic

---

## Execution Strategy

Single task, sequential execution.

---

## TODOs

- [x] 1. Separate singleton and embedding locks

  **What to do**:
  1. Open `backend/services/embedding_service.py`
  2. Line 14: Change `_lock = threading.Lock()` → `_init_lock = threading.Lock()`
  3. Line 18: Change `with cls._lock:` → `with cls._init_lock:`
  4. Line 21 (after `_initialized = False`): Add `cls._instance._embed_lock = threading.Lock()`
  5. Line 127: Change `with self._lock:` → `with self._embed_lock:`

  **Must NOT do**:
  - Don't remove any lock usage
  - Don't change singleton double-check pattern
  - Don't modify `_embed_batch` internals

  **Acceptance Criteria**:
  - [x] Syntax valid: `python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"`
  - [x] No deadlock: `python backend/scripts/ingest_pdfs.py --cap 1` completes

  **QA Scenarios**:
  ```
  Scenario: Cap 1 ingestion completes without deadlock
    Tool: Bash
    Steps:
      1. python backend/scripts/ingest_pdfs.py --cap 1
      2. Wait for completion (max 5 min)
      3. Check exit code is 0
      4. Grep output for "deadlock" - count must be 0
    Expected: Exit code 0, no "deadlock" in output
    Evidence: .sisyphus/evidence/deadlock-fix/cap1-ingestion.txt
  ```

  **Commit**: YES
  - Message: `fix(embedding): separate singleton and operation locks to prevent deadlock`
  - Files: `backend/services/embedding_service.py`

---

## Success Criteria

```bash
# Syntax check
python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"

# Ingestion test
python backend/scripts/ingest_pdfs.py --cap 1  # Expected: success, no deadlock
```
