# Fix TensorRT CUDA Runtime enqueueV3 Error

## TL;DR

> **Quick Summary**: TensorRT engine fails with CUDA Runtime error during enqueueV3 (shape copy H2D). Multiple singleton instances + corrupted cache + no cleanup between batches.
> 
> **Deliverables**:
> - Clear TensorRT cache before first use
> - Add cache cleanup utility
> - Verify singleton is truly single instance
> - Add proper error recovery with cache invalidation
> 
> **Estimated Effort**: Medium (20-30 mins)
> **Critical Path**: Clear cache → Verify singleton → Add recovery

---

## Context

### Error Message
```
[E:onnxruntime:Default, tensorrt_execution_provider.h:90] [2026-02-21 10:54:43 ERROR] 
IExecutionContext::enqueueV3: Error Code 1: Cuda Runtime 
(In nvinfer1::rt::cuda::doShapeCopyH2D at C:\_src\runtime\gpu\cuda\shapeHostToDeviceRunner.cpp:58)

RuntimeError: Embedding failed: TensorRT EP execution context enqueue failed.
```

### Symptoms
1. **Script keeps running** after error (doesn't stop on first failure)
2. **Duplicate initialization logs** (singleton pattern broken?)
3. **Random caps fail** (87, then continues to 89)
4. **Error is intermittent** (some caps work, others fail)

### Root Causes

#### 1. Corrupted TensorRT Cache
TensorRT engine was built with:
- Different CUDA context
- Different input shapes
- Different memory layout

Current inputs don't match cached engine expectations → crash during H2D copy.

#### 2. Singleton Pattern May Be Broken
Earlier logs showed:
```
[EmbeddingService] Building TensorRT engine... (appears TWICE)
[EmbeddingService] Active Providers: ['TensorrtExecutionProvider', ...]
[EmbeddingService] Active Providers: None  ← Second instance!
```

This suggests **two EmbeddingService instances** are being created, fighting over GPU resources.

#### 3. No Error Recovery
When TensorRT fails, the script:
- Catches exception ✓
- Prints error ✓
- **Continues to next cap** ✗ (should stop or rebuild engine)

---

## Work Objectives

### Core Objective
Make TensorRT embedding service robust against CUDA runtime errors through cache management and singleton verification.

### Concrete Deliverables
1. **Cache cleanup utility** - Clear TensorRT cache on demand
2. **Singleton verification** - Ensure only ONE instance exists
3. **Error recovery** - Invalidate cache and retry on TRT failures
4. **Ingestion script hardening** - Stop on embedding errors (don't continue silently)
5. **Torch CUDA installation** - Replace CPU-only torch with CUDA-enabled version (2.10.0+cu130)
6. **Disable TensorRT** - Use CUDA provider only (sm_120 incompatibility workaround)

### Definition of Done
- [ ] TensorRT cache can be cleared programmatically
- [ ] Singleton verified - only one instance logs "Building TensorRT engine..."
- [ ] On TensorRT error, cache is cleared and model reloaded
- [ ] Ingestion script stops immediately on embedding failure
- [ ] Torch 2.10.0+cu130 with CUDA support is installed (not CPU-only)
- [ ] TensorRT provider disabled, CUDA provider active
- [ ] Cap 1, 87, 89 all succeed consecutively

### Must Have
- Cache clear before first TensorRT load
- Single EmbeddingService instance across entire script run
- Hard stop on embedding errors (no silent failures)

### Must NOT Have
- Don't disable TensorRT (keep using it, just fix the errors)
- Don't add timeouts/retries without cache invalidation
- Don't suppress errors (fail fast and loud)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: Ad-hoc verification scripts
- **QA Policy**: Agent-executed scenarios with evidence capture

### QA Approach
- **Cache Management**: Bash - delete cache, verify rebuild
- **Singleton**: Python inline - count initialization logs
- **Error Recovery**: Bash - trigger error, verify cache clear + retry

---

## Execution Strategy

Sequential execution (dependencies between tasks).

```
Task 1: Add cache cleanup utility
  ↓
Task 2: Verify singleton is truly single
  ↓
Task 3: Add error recovery with cache invalidation
  ↓
Task 4: Harden ingestion script error handling
  ↓
Task 5: Fix torch CPU-only installation (install CUDA version)
  ↓
Task 6: Disable TensorRT provider (sm_120 incompatibility, use CUDA only)
```

---

## TODOs

- [x] 1. Add TensorRT cache cleanup utility

  **What to do**:
  1. Add method to `EmbeddingService`:
     ```python
     @classmethod
     def clear_tensorrt_cache(cls):
         """Clear TensorRT engine cache. Call before first load if issues occur."""
         import shutil
         cache_dir = os.path.join(
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "models", "yuan-onnx-trt", "cache"
         )
         if os.path.exists(cache_dir):
             print(f"[EmbeddingService] Clearing TensorRT cache: {cache_dir}")
             shutil.rmtree(cache_dir)
             os.makedirs(cache_dir, exist_ok=True)
             print("[EmbeddingService] Cache cleared.")
     ```
  2. Call `EmbeddingService.clear_tensorrt_cache()` at start of `ingest_pdfs.py` (before first `get_embedding_service()`)

  **Must NOT do**:
  - Don't clear cache on every embedding call (too slow)
  - Don't delete model files (only cache/ directory)

  **Acceptance Criteria**:
  - [x] `clear_tensorrt_cache()` method exists
  - [x] Called once at script startup
  - [x] Cache directory recreated empty

  **QA Scenarios**:
  ```
  Scenario: Cache is cleared before first TensorRT load
    Tool: Bash
    Steps:
      1. Create dummy cache file: echo "test" > backend/models/yuan-onnx-trt/cache/dummy.trt
      2. Run: python -c "from backend.services.embedding_service import EmbeddingService; EmbeddingService.clear_tensorrt_cache()"
      3. Check cache dir is empty: ls backend/models/yuan-onnx-trt/cache/
    Expected: Directory exists but is empty (no dummy.trt)
    Evidence: .sisyphus/evidence/trt-fix/cache-clear.txt
  ```

  **Commit**: NO (wait for all tasks)

---

- [x] 2. Verify singleton creates only one instance

  **What to do**:
  1. Add debug counter to `__new__`:
     ```python
     def __new__(cls):
         if cls._instance is None:
             with cls._init_lock:
                 if cls._instance is None:
                     print(f"[EmbeddingService] Creating NEW singleton instance (ID: {id(cls)})")
                     cls._instance = super(EmbeddingService, cls).__new__(cls)
                     cls._instance._initialized = False
                     cls._instance._embed_lock = threading.Lock()
         else:
             print(f"[EmbeddingService] Reusing existing singleton instance (ID: {id(cls._instance)})")
         return cls._instance
     ```
  2. Run test script that calls `get_embedding_service()` 3 times
  3. Verify logs show: "Creating NEW" once, "Reusing" twice

  **Must NOT do**:
  - Don't remove existing singleton logic
  - Don't add new class variables

  **Acceptance Criteria**:
  - [x] Only ONE "Creating NEW" log appears
  - [x] Instance ID is identical across all calls

  **QA Scenarios**:
  ```
  Scenario: Singleton returns same instance across multiple calls
    Tool: Bash (python inline)
    Steps:
      1. python -c "
         import sys; sys.path.insert(0, '.');
         from backend.core import setup_env; setup_env.setup_cuda_dlls();
         from backend.services.embedding_service import get_embedding_service;
         s1 = get_embedding_service();
         s2 = get_embedding_service();
         s3 = get_embedding_service();
         assert id(s1) == id(s2) == id(s3), 'Different instances!';
         print(f'All same: ID={id(s1)}');
         print('PASS')
         " 2>&1 | grep -E '(Creating|Reusing|PASS)'
      2. Expect: "Creating NEW" appears once, "Reusing" appears twice
    Expected: PASS
    Evidence: .sisyphus/evidence/trt-fix/singleton-verify.txt
  ```

  **Commit**: NO (wait for all tasks)

---

- [x] 3. Add error recovery with cache invalidation

  **What to do**:
  1. Wrap `_embed_batch` with TensorRT error detection:
     ```python
     def _embed_batch(self, texts: List[str]) -> List[List[float]]:
         try:
             # ... existing embedding logic ...
             return embeddings.cpu().numpy().tolist()
         except Exception as e:
             error_msg = str(e)
             if "TensorRT" in error_msg or "enqueueV3" in error_msg or "Cuda Runtime" in error_msg:
                 print("[EmbeddingService] TensorRT CUDA error detected. Clearing cache and failing.")
                 EmbeddingService.clear_tensorrt_cache()
                 raise RuntimeError(
                     f"TensorRT engine failed. Cache cleared. Please restart script. Original error: {error_msg}"
                 )
             raise
     ```
  2. This ensures:
     - TensorRT errors trigger cache clear
     - Script stops (doesn't continue with broken engine)
     - User knows to restart

  **Must NOT do**:
  - Don't auto-retry within same process (GPU state may be corrupted)
  - Don't clear cache on non-TRT errors

  **Acceptance Criteria**:
  - [x] TensorRT errors trigger cache clear
  - [x] Error message tells user to restart
  - [x] Non-TRT errors pass through unchanged

  **QA Scenarios**:
  ```
  Scenario: Simulated TRT error clears cache
    Tool: Bash (python inline - mock the error)
    Steps:
      1. Create test file that forces TRT error path
      2. Verify cache is cleared
      3. Verify exception is raised with "restart script" message
    Expected: Cache deleted, clear error message
    Evidence: .sisyphus/evidence/trt-fix/error-recovery.txt
  ```

  **Commit**: NO (wait for all tasks)

---

- [x] 4. Harden ingestion script error handling

  **What to do**:
  1. In `ingest_pdfs.py`, modify error handling:
     ```python
     except Exception as e:
         print(f"Error processing {cap}: {e}")
         traceback.print_exc()
         # ADD THIS:
         if "Embedding failed" in str(e):
             print("\n[CRITICAL] Embedding service failed. Stopping ingestion.")
             print("Possible causes:")
             print("  1. TensorRT cache corruption - try deleting backend/models/yuan-onnx-trt/cache/")
             print("  2. GPU memory exhausted - restart terminal")
             print("  3. CUDA driver issue - check nvidia-smi")
             sys.exit(1)  # HARD STOP on embedding errors
         # For non-embedding errors, continue (optional)
         continue
     ```
  2. This ensures embedding failures stop the script immediately

  **Must NOT do**:
  - Don't stop on parsing errors (those are fine to skip)
  - Don't suppress the traceback

  **Acceptance Criteria**:
  - [x] Embedding errors call `sys.exit(1)`
  - [x] Helpful diagnostic message printed
  - [x] Non-embedding errors can continue (optional)

  **QA Scenarios**:
  ```
  Scenario: Embedding error stops script with helpful message
    Tool: Bash
    Steps:
      1. Corrupt TRT cache to force error
      2. Run: python backend/scripts/ingest_pdfs.py --cap 1
      3. Verify exit code is 1 (not 0)
      4. Verify output contains "CRITICAL" and diagnostic steps
    Expected: Exit 1, clear error guidance
    Evidence: .sisyphus/evidence/trt-fix/hard-stop.txt
  ```

  **Commit**: NO (wait for all tasks)

---

- [x] 5. Fix torch CPU-only installation

  **What to do**:
  1. Check current torch installation:
     ```bash
     python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
     ```
  2. If torch is CPU-only (version ends with `+cpu`), reinstall with CUDA support:
     ```bash
     pip uninstall -y torch torchvision torchaudio
     pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
     ```
     **Note**: cu130 (CUDA 13.0) is required for sm_120 (RTX 5060 Ti) support.
     Older CUDA versions (cu118, cu124) don't support Blackwell architecture.
  3. Verify CUDA is available after installation:
     ```bash
     python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA OK')"
     ```

  **Must NOT do**:
  - Don't install older CUDA versions (cu118/cu124 don't support sm_120)
  - Don't skip verification step
  - Don't continue if CUDA is still unavailable

  **Acceptance Criteria**:
  - [x] torch version does NOT end with `+cpu` (now 2.10.0+cu130)
  - [x] `torch.cuda.is_available()` returns `True`
  - [x] torch recognizes RTX 5060 Ti with sm_120 (no warnings)

  **QA Scenarios**:
  ```
  Scenario: Torch with CUDA support is installed
    Tool: Bash
    Steps:
      1. python -c "import torch; print(torch.__version__)"
      2. Verify output does NOT contain "+cpu"
      3. python -c "import torch; print(torch.cuda.is_available())"
      4. Verify output is "True"
    Expected: Version shows cu130, CUDA available = True, no sm_120 warnings
    Evidence: .sisyphus/evidence/trt-fix/torch-cuda.txt
  ```

  **Commit**: NO (wait for Task 6)

---

- [ ] 6. Disable TensorRT provider (sm_120 incompatibility)

  **What to do**:
  1. **Root Cause**: ONNX Runtime 1.24.1 TensorRT doesn't support sm_120 (RTX 5060 Ti / Blackwell architecture)
     - PyTorch 2.10.0+cu130 works fine (no warnings)
     - TensorRT causes "CUDA failure 700: illegal memory access" during kernel execution
     - CUDA provider works without TensorRT optimization
  
  2. Modify `backend/services/embedding_service.py` (lines 78-134):
     - Remove TensorRT provider configuration
     - Remove TensorRT options (`trt_engine_cache_enable`, etc.)
     - Remove cache_path creation (no longer needed)
     - Change providers_config to: `["CUDAExecutionProvider", "CPUExecutionProvider"]`
     - Update loading logic to skip TensorRT entirely
     - Add explanatory comment about sm_120 incompatibility
  
  3. Specific changes:
     ```python
     # Replace lines 78-134 with:
     # 2. Configure Providers (Priority: CUDA -> CPU)
     # NOTE: TensorRT disabled due to sm_120 (RTX 5060 Ti) incompatibility.
     # ONNX Runtime 1.24.1 TensorRT doesn't support Blackwell architecture (sm_120).
     # PyTorch 2.10+cu130 works, but TensorRT causes "CUDA illegal memory access" errors.
     # Fallback to CUDA provider provides GPU acceleration without TensorRT optimization.
     sess_opt = ort.SessionOptions()
     sess_opt.intra_op_num_threads = 1
     sess_opt.inter_op_num_threads = 1

     providers_config = [
         "CUDAExecutionProvider",
         "CPUExecutionProvider",
     ]

     print(f"[EmbeddingService] Initializing ORT with providers: {providers_config}")
     print("[EmbeddingService] Note: TensorRT disabled (sm_120 incompatibility)")

     try:
         self.model = ORTModelForFeatureExtraction.from_pretrained(
             self.model_path,
             providers=providers_config,
             session_options=sess_opt,
             trust_remote_code=True,
         )
         print("[EmbeddingService] Loaded with CUDA provider.")
     except Exception as e:
         print(f"[EmbeddingService] CUDA load failed: {e}")
         print("[EmbeddingService] Falling back to CPU-only...")
         self.model = ORTModelForFeatureExtraction.from_pretrained(
             self.model_path,
             providers=["CPUExecutionProvider"],
             session_options=sess_opt,
             trust_remote_code=True,
         )
     ```

  **Must NOT do**:
  - Don't remove CUDA provider (it's our primary GPU acceleration now)
  - Don't remove the fallback to CPU (still needed for non-GPU environments)
  - Don't remove the comment explaining why TensorRT is disabled

  **Acceptance Criteria**:
  - [ ] TensorRT provider removed from providers_config
  - [ ] CUDA provider is first in list
  - [ ] Explanatory comment added about sm_120 incompatibility
  - [ ] Logs show "CUDA provider" instead of "TensorRT"

  **QA Scenarios**:
  ```
  Scenario: Embedding service uses CUDA without TensorRT
    Tool: Bash
    Steps:
      1. Clear any existing cache: rm -rf backend/models/yuan-onnx-trt/cache/*
      2. Run: python -c "
         from backend.core import setup_env; setup_env.setup_cuda_dlls();
         from backend.services.embedding_service import get_embedding_service;
         svc = get_embedding_service();
         result = svc.embed_query('test');
         print(f'Embedding length: {len(result)}');
         print('PASS')
         " 2>&1 | grep -E '(Initializing ORT|Active Providers|PASS)'
      3. Verify output shows "CUDAExecutionProvider" (NOT "TensorrtExecutionProvider")
      4. Verify output shows "PASS" (embedding succeeded)
    Expected: CUDA provider active, no TensorRT, embedding works
    Evidence: .sisyphus/evidence/trt-fix/cuda-only.txt
  ```

  **Commit**: YES (all tasks complete)
  - Message: `fix(embedding): disable TensorRT for sm_120, upgrade torch to 2.10+cu130, add cache mgmt`
  - Files: `backend/services/embedding_service.py`, `backend/scripts/ingest_pdfs.py`
  - Pre-commit: Verify embedding service initializes without errors

---

## Final Verification Wave

- [ ] F1. **Full ingestion test** — Run caps 1, 87, 89 consecutively

  ```bash
  # Clear cache first
  rm -rf backend/models/yuan-onnx-trt/cache/*
  
  # Run all problem caps
  for cap in 1 87 89; do
    echo "Testing Cap $cap..."
    python backend/scripts/ingest_pdfs.py --cap $cap || exit 1
  done
  ```
  
  Expected: All caps complete successfully, no TensorRT errors
  
  Output: `All 3 caps: PASS | FAIL (which cap)`

---

## Success Criteria

### Verification Commands
```bash
# 1. Cache clear works
python -c "from backend.services.embedding_service import EmbeddingService; EmbeddingService.clear_tensorrt_cache()"

# 2. Singleton verified
python -c "from backend.services.embedding_service import get_embedding_service; [get_embedding_service() for _ in range(3)]" 2>&1 | grep -c "Creating NEW"  # Should be 1

# 3. Ingestion succeeds
python backend/scripts/ingest_pdfs.py --cap 1 87 89
```

### Final Checklist
- [ ] Cache cleanup utility added
- [ ] Singleton verified (only 1 instance)
- [ ] TensorRT error recovery implemented
- [ ] Ingestion script stops on embedding errors
- [ ] Torch 2.10.0+cu130 installed (not CPU-only)
- [ ] TensorRT disabled, CUDA provider active
- [ ] Caps 1, 87, 89 all succeed
