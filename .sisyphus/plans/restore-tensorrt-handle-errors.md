# Restore TensorRT Acceleration with Error Handling

## TL;DR

> **Quick Summary**: Restore TensorRT GPU acceleration (was incorrectly disabled) and add graceful error handling for occasional `IExecutionContext::enqueueV3` failures.
> 
> **Deliverables**:
> - TensorRT enabled as primary provider
> - Retry logic for transient TensorRT errors
> - Cache invalidation on persistent failures
> - Clean terminal output (no spam)
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - single file fix
> **Critical Path**: Restore providers → Add retry → Test

---

## Context

### Original Request
User reported TensorRT errors spamming terminal during `python backend/scripts/ingest_pdfs.py` runs.

### What Went Wrong
**I misdiagnosed the problem** and disabled GPU entirely:
- Thought: "TensorRT doesn't support sm_120"
- Reality: TensorRT works fine, just has occasional transient errors
- Result: Switched to CPU-only mode (3-5x slower)

### Actual Problem
TensorRT **works** but occasionally throws:
```
[E:onnxruntime] IExecutionContext::enqueueV3: Error Code 1: Cuda Runtime
TensorRT EP execution context enqueue failed.
```

**These are transient errors** - retrying usually works.

---

## Work Objectives

### Core Objective
Restore TensorRT GPU acceleration with graceful error handling for transient failures.

### Concrete Deliverables
1. `backend/services/embedding_service.py` - TensorRT enabled, retry logic added
2. Terminal output clean (errors handled, not spammed)
3. Ingestion succeeds without manual intervention

### Definition of Done
- [ ] TensorRT is primary provider (confirmed in logs)
- [ ] Embedding errors auto-retry (up to 3 attempts)
- [ ] Persistent failures trigger cache clear + fallback
- [ ] Run `python backend/scripts/ingest_pdfs.py --cap 1 87 89` - all succeed
- [ ] Terminal shows clean progress (no error spam)

### Must Have
- TensorRT enabled as primary provider
- Retry logic (3 attempts with exponential backoff)
- Cache invalidation on persistent failures
- Fallback to CUDA provider if TensorRT unusable

### Must NOT Have (Guardrails)
- Don't disable TensorRT/GPU (that was the mistake)
- Don't suppress errors silently (log them, then handle)
- Don't retry indefinitely (max 3 attempts)
- Don't clear cache on every error (only after 3 failures)

---

## TODOs

- [ ] 1. Restore TensorRT provider configuration

  **What to do**:
  1. Open `backend/services/embedding_service.py`
  2. Replace lines 78-109 (CPU-only config) with TensorRT config:
     ```python
     # 2. Configure Providers (Priority: TensorRT -> CUDA -> CPU)
     sess_opt = ort.SessionOptions()
     sess_opt.intra_op_num_threads = 1
     sess_opt.inter_op_num_threads = 1
     
     cache_path = os.path.join(self.model_path, "cache")
     os.makedirs(cache_path, exist_ok=True)
     
     trt_options = {
         "trt_engine_cache_enable": True,
         "trt_engine_cache_path": cache_path,
         "trt_fp16_enable": True,
         "trt_detailed_build_log": False,
     }
     
     providers_config = [
         ("TensorrtExecutionProvider", trt_options),
         "CUDAExecutionProvider",
         "CPUExecutionProvider",
     ]
     
     print(f"[EmbeddingService] Initializing ORT with TensorRT acceleration")
     print("[EmbeddingService] Building TensorRT engine... (5-15 mins on first run)")
     
     self.model = ORTModelForFeatureExtraction.from_pretrained(
         self.model_path,
         providers=providers_config,
         session_options=sess_opt,
         trust_remote_code=True,
     )
     ```
  
  **Must NOT do**:
  - Don't use CPU-only config
  - Don't remove TensorRT options
  - Don't remove cache path setup

  **Acceptance Criteria**:
  - [ ] Line 96 uses TensorRT provider (not CPU)
  - [ ] Logs show "TensorRT acceleration" on startup
  - [ ] Active Providers includes `TensorrtExecutionProvider`

  **QA Scenarios**:
  ```
  Scenario: TensorRT loads successfully
    Tool: Bash
    Steps:
      1. Run: python -c "
         from backend.core import setup_env; setup_env.setup_cuda_dlls();
         from backend.services.embedding_service import get_embedding_service;
         svc = get_embedding_service();
         print('Active providers:', svc.model.model.get_providers());
         " 2>&1 | grep -E "(TensorRT|Active providers)"
      2. Verify output contains "TensorrtExecutionProvider"
    Expected: TensorRT is active provider
    Evidence: .sisyphus/evidence/tensorrt-restore/provider-check.txt
  ```

  **Commit**: NO (wait for all tasks)

---

- [ ] 2. Add retry logic to `_embed_batch` method

  **What to do**:
  1. Find `_embed_batch` method (around line 155)
  2. Wrap embedding execution in retry loop:
     ```python
     def _embed_batch(self, texts: List[str]) -> List[List[float]]:
         """Generate embeddings with retry on transient TensorRT errors."""
         MAX_RETRIES = 3
         RETRY_DELAY = [0.5, 1.0, 2.0]  # Exponential backoff
         
         for attempt in range(MAX_RETRIES):
             try:
                 # Tokenize
                 inputs = self.tokenizer(
                     texts,
                     padding=True,
                     truncation=True,
                     max_length=512,
                     return_tensors="pt",
                 )
                 
                 # Run inference
                 with self._embed_lock:
                     outputs = self.model(**inputs)
                 
                 # Mean pooling
                 embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
                 embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
                 
                 # Convert to list
                 result = embeddings_normalized.cpu().numpy().tolist()
                 
                 # Validate (catch all-zero bug)
                 if all(all(abs(x) < 1e-6 for x in emb) for emb in result):
                     raise RuntimeError("All-zero embeddings detected (model broken)")
                 
                 return result
             
             except Exception as e:
                 error_msg = str(e)
                 is_transient = "enqueueV3" in error_msg or "TensorRT EP" in error_msg
                 
                 if is_transient and attempt < MAX_RETRIES - 1:
                     print(f"[EmbeddingService] TensorRT error (attempt {attempt+1}/{MAX_RETRIES}): {error_msg}")
                     print(f"[EmbeddingService] Retrying in {RETRY_DELAY[attempt]}s...")
                     time.sleep(RETRY_DELAY[attempt])
                     continue
                 else:
                     # Persistent failure - clear cache and re-raise
                     print(f"[EmbeddingService] PERSISTENT ERROR after {attempt+1} attempts")
                     print(f"[EmbeddingService] Clearing TensorRT cache...")
                     self.clear_tensorrt_cache()
                     raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries: {error_msg}")
     ```
  3. Add `import time` at top of file

  **Must NOT do**:
  - Don't retry indefinitely
  - Don't clear cache on first error
  - Don't suppress error messages

  **Acceptance Criteria**:
  - [ ] Retry loop handles transient errors
  - [ ] Max 3 attempts before failure
  - [ ] Exponential backoff (0.5s, 1s, 2s)
  - [ ] Cache cleared only after 3 failures

  **QA Scenarios**:
  ```
  Scenario: Transient error retries successfully
    Tool: Bash (manual simulation not feasible - rely on production behavior)
    Steps:
      1. Run ingestion for multiple caps: python backend/scripts/ingest_pdfs.py --cap 1 87 89
      2. Monitor logs for retry messages
      3. Verify no crashes despite occasional TensorRT errors
    Expected: Ingestion completes, retries visible in logs
    Evidence: .sisyphus/evidence/tensorrt-restore/retry-logs.txt
  ```

  **Commit**: NO (wait for final task)

---

- [ ] 3. Remove incorrect comments about sm_120 incompatibility

  **What to do**:
  1. Find lines 79-91 (sm_120 incompatibility comments)
  2. Delete entirely (this was incorrect diagnosis)
  3. Lines 98-102 (CPU-only warnings) - delete (replaced by TensorRT logs)

  **Must NOT do**:
  - Don't leave outdated comments
  - Don't mention CPU-only mode

  **Acceptance Criteria**:
  - [ ] No mentions of "sm_120 incompatibility"
  - [ ] No mentions of "CPU-only mode"
  - [ ] Clean provider configuration code

  **Commit**: YES
  - Message: `fix(embedding): restore TensorRT with retry logic for transient errors`
  - Files: `backend/services/embedding_service.py`
  - Pre-commit: `python -c "from backend.services.embedding_service import get_embedding_service; get_embedding_service()"`

---

## Success Criteria

### Verification Commands
```bash
# 1. TensorRT loads
python -c "from backend.core import setup_env; setup_env.setup_cuda_dlls(); from backend.services.embedding_service import get_embedding_service; svc = get_embedding_service(); print(svc.model.model.get_providers())"

# 2. Ingestion succeeds with retry handling
python backend/scripts/ingest_pdfs.py --cap 1 87 89
```

### Final Checklist
- [ ] TensorRT is active provider (logs confirm)
- [ ] Retry logic handles transient errors (no crashes)
- [ ] Ingestion completes successfully for caps 1, 87, 89
- [ ] Terminal output is clean (errors logged, not spammed)
- [ ] No CPU-only mode comments remain
