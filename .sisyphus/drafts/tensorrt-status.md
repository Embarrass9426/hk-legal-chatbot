# TensorRT and CUDA Status in WSL2

**Date**: 2026-02-15
**Status**: ❌ GPU Acceleration Unavailable (Running on CPU)

## Current Behavior
The ingestion script defaults to CPU execution despite `TensorrtExecutionProvider` and `CUDAExecutionProvider` being requested.

### Log Output
```
[EmbeddingService] Initializing ORT with providers: [('TensorrtExecutionProvider', ...), 'CUDAExecutionProvider', 'CPUExecutionProvider']
[EmbeddingService] TensorRT load failed: ... available execution providers are ['AzureExecutionProvider', 'CPUExecutionProvider'].
[EmbeddingService] Falling back to CUDA/CPU...
[EmbeddingService] CUDA load failed: ... available execution providers are ['AzureExecutionProvider', 'CPUExecutionProvider'].
[EmbeddingService] Active Providers: ['CPUExecutionProvider']
```

## Diagnosis
The `onnxruntime-gpu` package in the environment does not detect the WSL2 GPU drivers, even though `LD_LIBRARY_PATH` was configured in `.bashrc`.

Possible causes:
1. `onnxruntime` vs `onnxruntime-gpu` conflict (both might be installed)
2. Missing `libcudnn` or other specific libraries in WSL2
3. ONNX Runtime version mismatch with CUDA version (Windows host driver vs WSL2 libs)

## Next Steps (Recommended)
1. Verify installed packages: `pip list | grep onnxruntime`
2. If `onnxruntime` (CPU) is present, uninstall it and force reinstall `onnxruntime-gpu`
3. Verify WSL2 GPU visibility via `nvidia-smi` inside WSL2 terminal
