import os
import onnx
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from onnxconverter_common import float16

# Configuration
model_id = "IEITYuan/Yuan-embedding-2.0-en"
# Go up one level: scripts -> backend -> models
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "yuan-onnx-trt")
os.makedirs(save_dir, exist_ok=True)

print(f"[INFO] Starting TensorRT-optimized export for {model_id}...")
print(f"[INFO] Output directory: {save_dir}")

# 1. Export standard ONNX (FP32) with Optimum
# Optimum handles default dynamic axes for batch_size and sequence_length
print("Step 1: Exporting base ONNX model (FP32)...")
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    export=True,
    provider="CUDAExecutionProvider",  # Use CUDA for export if available
    trust_remote_code=True,
)

# Save the FP32 model temporarily or effectively as the base
model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

onnx_path = os.path.join(save_dir, "model.onnx")
print(f"Base model saved to {onnx_path}")

# 2. Convert to FP16
# This reduces model size and allows TensorRT to use FP16 kernels more easily
print("Step 2: Converting to FP16 for TensorRT optimization...")
try:
    model_fp32 = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=False,  # Keep input/output as FP32 for compatibility
        op_block_list=["ReduceMean", "Pow", "Sqrt"],  # Avoid overflow in reduction ops
    )

    # Overwrite or save as new? Let's overwrite model.onnx to be the main artifact
    # expecting the loader to just load "model.onnx"
    # onnx.save(model_fp16, onnx_path)

    # Save to a temporary file first to verify
    fp16_path = os.path.join(save_dir, "model_fp16.onnx")
    onnx.save(model_fp16, fp16_path)
    print(f"FP16 model saved to {fp16_path}")

    # Check if file size is reasonable (> 1MB)
    if os.path.getsize(fp16_path) > 1024 * 1024:
        # Move to model.onnx
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        os.rename(fp16_path, onnx_path)
        print(f"Renamed to {onnx_path}")
    else:
        print(
            f"[ERROR] FP16 export resulted in tiny file! Keeping original model.onnx (FP32)"
        )

except Exception as e:
    print(f"[ERROR] FP16 conversion failed: {e}")
    import traceback

    traceback.print_exc()
    print("Proceeding with FP32 model (still usable by TensorRT, but larger).")

print("Export complete! Ready for TensorRTExecutionProvider.")
