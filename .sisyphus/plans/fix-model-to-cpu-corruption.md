# Fix model.to("cpu") Corrupting ONNX Session

## TL;DR

> **Quick Summary**: Line 119 `self.model = self.model.to("cpu")` destroys the ONNX session, causing "Active Providers: None" and hanging embeddings.
> 
> **Deliverables**:
> - Remove line 119 from `backend/services/embedding_service.py`
> - Verify ONNX session persists and shows correct providers
> 
> **Estimated Effort**: 2 minutes (delete 1 line)
> **Critical Path**: Delete line → Test

---

## Context

### Error Symptoms
```
[EmbeddingService] Loaded with TensorRT options.
[EmbeddingService] Active Providers: None  ← Should show ['TensorrtExecutionProvider', 'CPUExecutionProvider']
Generating embeddings for 90 / 90 chunks in Cap 1...
[HANGS FOREVER]
```

### Root Cause
Line 119: `self.model = self.model.to("cpu")` **reassigns** `self.model` to a new object that:
- Lacks the `.model` attribute (ONNX InferenceSession)
- Cannot run inference
- Shows `Active Providers: None` because session is destroyed

### Why This Line Exists
Added during previous fix to force CPU mode for PyTorch operations. **This was wrong** - ONNX Runtime doesn't need explicit `.to()` calls.

---

## Work Objectives

### Core Objective
Remove the corrupting `.to("cpu")` call that destroys the ONNX session.

### Concrete Deliverables
- Line 119 deleted from `backend/services/embedding_service.py`
- ONNX session persists with correct providers
- Embeddings generate successfully

### Definition of Done
- [ ] Line 119 removed
- [ ] `hasattr(self.model, "model")` returns True after loading
- [ ] Active Providers shows TensorRT in logs
- [ ] Cap 1 ingestion completes without hanging

### Must Have
- ONNX session preserved after model loading
- TensorRT provider active and functional

### Must NOT Have
- No `.to()` calls on the model object
- No other changes to loading logic

---

## Execution Strategy

Single file, single line deletion. Sequential execution.

---

## TODOs

- [ ] 1. Remove model.to("cpu") call

  **What to do**:
  1. Open `backend/services/embedding_service.py`
  2. Delete line 119: `self.model = self.model.to("cpu")`
  3. Save file

  **Must NOT do**:
  - Don't add any device placement logic back
  - Don't modify provider loading (lines 83-110)
  - Don't touch the provider logging (lines 112-117)

  **Acceptance Criteria**:
  - [ ] Syntax valid: `python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"`
  - [ ] Line 119 no longer contains `.to("cpu")`

  **QA Scenarios**:
  ```
  Scenario: ONNX session persists after loading
    Tool: Bash (python inline)
    Steps:
      1. python -c "
         import sys; sys.path.insert(0, '.');
         from backend.core import setup_env; setup_env.setup_cuda_dlls();
         from backend.services.embedding_service import get_embedding_service;
         service = get_embedding_service();
         has_session = hasattr(service.model, 'model');
         providers = service.model.model.get_providers() if has_session else None;
         print(f'Has session: {has_session}');
         print(f'Providers: {providers}');
         assert has_session, 'ONNX session destroyed!';
         assert 'TensorrtExecutionProvider' in providers, 'TensorRT not active!';
         print('PASS')
         "
      2. Expect output: Has session: True, Providers: ['TensorrtExecutionProvider', ...]
    Expected: PASS
    Evidence: .sisyphus/evidence/fix-model-corruption/session-check.txt

  Scenario: Cap 1 ingestion completes without hanging
    Tool: Bash
    Steps:
      1. timeout 120s python backend/scripts/ingest_pdfs.py --cap 1
      2. Check exit code is 0 (not timeout 124)
      3. Grep output for "Upserted batch" to confirm completion
    Expected: Exit code 0, batch upserted
    Evidence: .sisyphus/evidence/fix-model-corruption/cap1-full.txt
  ```

  **Commit**: YES
  - Message: `fix(embedding): remove model.to(cpu) that corrupts ONNX session`
  - Files: `backend/services/embedding_service.py`

---

## Success Criteria

```bash
# 1. Syntax valid
python -c "import ast; ast.parse(open('backend/services/embedding_service.py', encoding='utf-8').read())"

# 2. Session persists
python -c "from backend.services.embedding_service import get_embedding_service; s = get_embedding_service(); assert hasattr(s.model, 'model'); print('PASS')"

# 3. Ingestion works
python backend/scripts/ingest_pdfs.py --cap 1  # Should complete in < 2 mins
```
