# Restore TensorRT Provider Configuration

## TL;DR

> **Quick Summary**: Fix `embedding_service.py` by restoring simple TensorRT provider configuration that matches the working `ingest_legal_pdfs.py` implementation.
> 
> **Deliverables**: 
> - TensorRT enabled as primary provider in embedding_service.py
> - GPU acceleration working (verified via Task Manager)
> - No terminal spam during ingestion
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - sequential tasks
> **Critical Path**: Task 1 → Task 2 → Task 3

---

## Context

### Original Request
User reported that embeddings were generating successfully with TensorRT but had occasional terminal spam. I mistakenly disabled TensorRT in commit `d8132c5`, thinking there was an sm_120 incompatibility issue.

### Investigation Findings
1. **User's GPU WAS working**: Task Manager showed consistent 9-10GB VRAM usage during ingestion
2. **TensorRT was the active provider**: User saw "Using TensorRT provider" in logs and stable memory usage (not the accumulation pattern that CUDA shows)
3. **Terminal spam was just noise**: Not actual failures - embeddings were generating successfully
4. **I broke it**: Removed TensorRT from providers list in my "fix" attempt

### Current Working Configuration
`ingest_legal_pdfs.py` (lines 305-314) has the correct, simple configuration:
```python
model = ORTModelForFeatureExtraction.from_pretrained(
    model_path,
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
    session_options=sess_opt,
    trust_remote_code=True,
)
```

This is what we need to restore in `embedding_service.py`.

---

## Work Objectives

### Core Objective
Restore TensorRT provider configuration in `embedding_service.py` to match the working implementation in `ingest_legal_pdfs.py`.

### Concrete Deliverables
- `backend/services/embedding_service.py` with TensorRT as primary provider
- Test script output showing TensorRT is active provider
- Successful embedding generation without errors

### Definition of Done
- [ ] `embedding_service.py` providers list includes TensorRT first: `["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]`
- [ ] Test script confirms active provider is TensorRT
- [ ] GPU usage visible in Task Manager during embedding generation
- [ ] No terminal error spam during test run

### Must Have
- TensorRT as first provider in list
- Keep retry logic (it's still useful for transient errors)
- Keep cache management utility (useful for troubleshooting)

### Must NOT Have (Guardrails)
- Don't remove retry logic or cache management - those are good additions
- Don't add complex provider options dictionary unless needed
- Don't change the session options (intra_op_num_threads, inter_op_num_threads)
- Don't suppress errors silently - let them surface if they occur

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (no pytest setup)
- **Automated tests**: None
- **Verification**: Manual test scripts with agent-executed QA scenarios

### QA Policy
Every task includes agent-executed QA scenarios with evidence saved to `.sisyphus/evidence/tensorrt-restore/`.

- **Python module test**: Use Bash (python) - Import, call functions, check output
- **GPU verification**: Use Bash (Task Manager via PowerShell) - Query GPU memory usage

---

## Execution Strategy

### Sequential Execution (No Parallelism)

```
Task 1: Restore TensorRT provider configuration
  ↓
Task 2: Test embedding generation with verification
  ↓
Task 3: Full integration test with ingest_pdfs.py
```

**Why Sequential**: Each task depends on the previous one completing successfully. No benefit to parallel execution for this simple fix.

---

## TODOs

- [ ] 1. Restore TensorRT Provider Configuration

  **What to do**:
  - Open `backend/services/embedding_service.py`
  - Locate the `_load_model()` method (around line 67-95)
  - Find the providers_config assignment (currently line 84)
  - Change from:
    ```python
    providers_config = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ```
  - To:
    ```python
    providers_config = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    ```
  - Update the print statement (line 86-88) to reflect TensorRT being enabled:
    ```python
    print("[EmbeddingService] Initializing ORT with TensorRT provider")
    ```
  - Keep all other code unchanged (retry logic, cache management, session options)

  **Must NOT do**:
  - Don't add provider_options dictionary (keep it simple like ingest_legal_pdfs.py)
  - Don't remove retry logic in `_embed_batch()` method
  - Don't remove `clear_tensorrt_cache()` classmethod
  - Don't change session options

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple configuration change in single file, well-defined edit
  - **Skills**: None needed
    - Reason: Straightforward code edit, no specialized tools required

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (must complete before Task 2)
  - **Blocks**: Task 2 (verification depends on this change)
  - **Blocked By**: None (can start immediately)

  **References**:
  
  **Current file to modify**:
  - `backend/services/embedding_service.py:79-95` - Current provider config (CUDA-only, needs TensorRT added)
  
  **Working reference implementation**:
  - `backend/ingest_legal_pdfs.py:305-314` - Working TensorRT provider config pattern to copy
  
  **WHY each reference matters**:
  - `ingest_legal_pdfs.py` shows the exact working configuration that user has been using successfully - copy this pattern exactly
  - The provider list should be simple (no options dictionary) to match the working implementation

  **Acceptance Criteria**:
  
  **Code changes verified**:
  - [ ] `providers_config` list includes `"TensorrtExecutionProvider"` as first element
  - [ ] Print statement updated to reflect TensorRT being enabled
  - [ ] Retry logic in `_embed_batch()` method unchanged (lines 120-208)
  - [ ] `clear_tensorrt_cache()` method unchanged (lines 50-65)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify TensorRT is in providers list
    Tool: Bash (python)
    Preconditions: embedding_service.py has been edited
    Steps:
      1. cd backend && python -c "from services.embedding_service import EmbeddingService; svc = EmbeddingService(); print('Active providers:', svc.model.model.get_providers())"
      2. Check output contains "TensorrtExecutionProvider"
      3. Verify it appears BEFORE "CPUExecutionProvider" in the list
    Expected Result: Output shows ['TensorrtExecutionProvider', ...other providers] with TensorRT first
    Failure Indicators: "CPUExecutionProvider" only in list, or TensorRT not present
    Evidence: .sisyphus/evidence/tensorrt-restore/task-1-provider-list.txt

  Scenario: Verify code compiles and imports successfully
    Tool: Bash (python)
    Preconditions: embedding_service.py edited
    Steps:
      1. cd backend && python -c "from services.embedding_service import EmbeddingService; print('Import successful')"
      2. Check exit code is 0
      3. Check no syntax errors or import errors in output
    Expected Result: "Import successful" printed, exit code 0
    Failure Indicators: SyntaxError, ImportError, or non-zero exit code
    Evidence: .sisyphus/evidence/tensorrt-restore/task-1-import-test.txt
  ```

  **Evidence to Capture**:
  - [ ] task-1-provider-list.txt - Active providers output
  - [ ] task-1-import-test.txt - Import test result

  **Commit**: YES
  - Message: `fix(embedding): restore TensorRT provider configuration`
  - Files: `backend/services/embedding_service.py`
  - Pre-commit: `cd backend && python -c "from services.embedding_service import EmbeddingService"`

---

- [ ] 2. Test Embedding Generation with TensorRT

  **What to do**:
  - Create a test script `backend/tests/test_tensorrt_embedding.py`
  - Script should:
    1. Import EmbeddingService
    2. Create instance (triggers model load with TensorRT)
    3. Generate embeddings for sample texts
    4. Print active providers
    5. Verify embeddings are non-zero
    6. Check GPU memory usage via PowerShell
  - Run the script and capture output
  - Verify TensorRT is active and GPU memory increases during embedding

  **Must NOT do**:
  - Don't use CPU-only mode
  - Don't skip provider verification
  - Don't ignore GPU memory check

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple test script creation and execution
  - **Skills**: None needed
    - Reason: Standard Python testing, no specialized tools

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (must wait for Task 1)
  - **Blocks**: Task 3 (full integration test)
  - **Blocked By**: Task 1 (needs TensorRT config restored)

  **References**:
  
  **Test pattern reference**:
  - `backend/tests/test_embedding_similarity.py` - Example test script structure (if exists)
  - `backend/services/embedding_service.py:104-118` - Public API methods to test (embed_documents, embed_query)
  
  **GPU monitoring reference**:
  - PowerShell command: `(Get-Process -Name python | Get-Counter '\Process(python*)\Working Set - Private').CounterSamples | Select -First 1`
  - Or simpler: Check Task Manager manually during test run
  
  **WHY each reference matters**:
  - embedding_service.py shows the public API to call in test
  - GPU monitoring confirms TensorRT is actually using GPU, not falling back to CPU

  **Acceptance Criteria**:
  
  **Test script created and runs**:
  - [ ] `backend/tests/test_tensorrt_embedding.py` exists
  - [ ] Script imports EmbeddingService successfully
  - [ ] Script generates embeddings for sample texts without errors

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Generate embeddings and verify TensorRT active
    Tool: Bash (python)
    Preconditions: Task 1 complete, embedding_service.py has TensorRT config
    Steps:
      1. cd backend && python tests/test_tensorrt_embedding.py
      2. Verify output contains "Active Providers: ['TensorrtExecutionProvider'"
      3. Verify embeddings are printed and non-zero
      4. Check no error messages or exceptions in output
    Expected Result: Script completes successfully, TensorRT shown as active, embeddings generated
    Failure Indicators: "CPUExecutionProvider" only, errors during embedding, all-zero vectors
    Evidence: .sisyphus/evidence/tensorrt-restore/task-2-embedding-test.txt

  Scenario: Verify GPU memory usage during embedding
    Tool: Bash (PowerShell via python subprocess)
    Preconditions: test_tensorrt_embedding.py running
    Steps:
      1. Before running test: Note GPU memory baseline
      2. Run: cd backend && python tests/test_tensorrt_embedding.py
      3. During execution: Open Task Manager → Performance → GPU
      4. Observe GPU memory usage increases (9-10GB as user reported)
    Expected Result: GPU memory shows allocation during embedding generation
    Failure Indicators: GPU memory stays flat (indicates CPU fallback)
    Evidence: .sisyphus/evidence/tensorrt-restore/task-2-gpu-usage.txt (manual observation notes)
  ```

  **Evidence to Capture**:
  - [ ] task-2-embedding-test.txt - Test script output with provider info
  - [ ] task-2-gpu-usage.txt - GPU memory observation notes

  **Commit**: YES
  - Message: `test(embedding): add TensorRT verification test`
  - Files: `backend/tests/test_tensorrt_embedding.py`
  - Pre-commit: `cd backend && python tests/test_tensorrt_embedding.py`

---

- [ ] 3. Full Integration Test with ingest_pdfs.py

  **What to do**:
  - Run `python backend/scripts/ingest_pdfs.py --cap 282 --embedding-batch 128` (or a small Cap number)
  - Monitor terminal output for:
    1. "Using TensorRT provider" message
    2. Successful embedding generation messages
    3. Successful upload to Pinecone
    4. **Absence of terminal spam** (no repeated error messages)
  - Monitor Task Manager to confirm GPU usage (9-10GB VRAM as user reported)
  - Capture full terminal output

  **Must NOT do**:
  - Don't run with --skip-upload (we want to verify full pipeline)
  - Don't use high concurrency initially (start with default batch size)
  - Don't ignore terminal warnings if they appear

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Running existing script with monitoring, no code changes
  - **Skills**: None needed
    - Reason: Standard script execution and observation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (must wait for Task 2)
  - **Blocks**: None (final verification task)
  - **Blocked By**: Task 2 (needs confirmed working embedding service)

  **References**:
  
  **Script to run**:
  - `backend/scripts/ingest_pdfs.py` - Main ingestion script (or `backend/ingest_legal_pdfs.py` if that's the one)
  - Lines 14-17: TensorRT environment variables already set
  - Lines 305-318: Model loading with TensorRT provider
  
  **Expected behavior**:
  - User's previous experience: "embedding generation and uploading successful", GPU usage 9-10GB, occasional terminal spam (which we want to eliminate)
  
  **WHY references matter**:
  - ingest_pdfs.py is the real-world workload user runs - this is the ultimate integration test
  - User's reported behavior gives us baseline to compare against

  **Acceptance Criteria**:
  
  **Successful ingestion run**:
  - [ ] Script completes without exceptions
  - [ ] Terminal shows "Using TensorRT provider" (or similar confirmation)
  - [ ] Embeddings generated successfully for all chunks
  - [ ] Vectors uploaded to Pinecone successfully
  - [ ] **No terminal spam** (no repeated error messages flooding output)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Run ingestion for single Cap with TensorRT
    Tool: Bash (python)
    Preconditions: Tasks 1 & 2 complete, TensorRT confirmed working
    Steps:
      1. cd backend && python scripts/ingest_pdfs.py --cap 282 --batch 1 --embedding-batch 64
      2. Monitor terminal output for provider confirmation message
      3. Wait for completion (should show "Pipeline complete!" message)
      4. Count number of error/warning lines in output
      5. Verify exit code is 0
    Expected Result: Script completes successfully, TensorRT confirmed, 0 or minimal warnings (not spam)
    Failure Indicators: Repeated error messages (spam pattern), exit code non-zero, "CPUExecutionProvider" only
    Evidence: .sisyphus/evidence/tensorrt-restore/task-3-ingestion-output.txt

  Scenario: Verify GPU usage during ingestion
    Tool: Interactive observation (Task Manager)
    Preconditions: ingest_pdfs.py starting
    Steps:
      1. Open Task Manager → Performance → GPU 0
      2. Note baseline GPU memory usage
      3. Start ingestion script
      4. Observe GPU memory increases to 9-10GB range (as user reported)
      5. Observe memory stays stable (not accumulating like CUDA mode)
    Expected Result: GPU memory shows 9-10GB usage during ingestion, stable pattern
    Failure Indicators: GPU memory stays at baseline (CPU fallback), or memory keeps growing (CUDA mode leak)
    Evidence: .sisyphus/evidence/tensorrt-restore/task-3-gpu-observation.txt (manual notes)

  Scenario: Verify no terminal spam (error flood check)
    Tool: Bash (grep)
    Preconditions: Ingestion output saved to file
    Steps:
      1. grep -i "error\|fail\|illegal" .sisyphus/evidence/tensorrt-restore/task-3-ingestion-output.txt | wc -l
      2. Count should be 0 or very low (< 5 lines)
      3. If errors found, check they are unique (not repeated spam)
    Expected Result: Zero or minimal error messages, no repeated spam pattern
    Failure Indicators: High error count (> 10), or same error repeated many times
    Evidence: .sisyphus/evidence/tensorrt-restore/task-3-error-count.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-ingestion-output.txt - Full terminal output from ingestion run
  - [ ] task-3-gpu-observation.txt - GPU memory usage notes
  - [ ] task-3-error-count.txt - Error line count from output

  **Commit**: NO (no code changes, just verification)

---

## Final Verification Wave

> All verification is agent-executed within the task QA scenarios above. No separate final verification wave needed for this quick fix.

---

## Commit Strategy

- **Task 1**: `fix(embedding): restore TensorRT provider configuration` — `backend/services/embedding_service.py`
- **Task 2**: `test(embedding): add TensorRT verification test` — `backend/tests/test_tensorrt_embedding.py`

---

## Success Criteria

### Verification Commands
```bash
# Verify TensorRT in providers
cd backend && python -c "from services.embedding_service import EmbeddingService; svc = EmbeddingService(); print(svc.model.model.get_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# Run test
cd backend && python tests/test_tensorrt_embedding.py
# Expected: Script completes, shows TensorRT active, embeddings generated

# Full ingestion test
cd backend && python scripts/ingest_pdfs.py --cap 282 --batch 1
# Expected: Completes successfully, GPU usage visible, no terminal spam
```

### Final Checklist
- [ ] TensorRT is first provider in embedding_service.py
- [ ] Test script confirms TensorRT active
- [ ] GPU memory usage visible during embedding (9-10GB)
- [ ] Ingestion completes without terminal spam
- [ ] All evidence files captured in .sisyphus/evidence/tensorrt-restore/
