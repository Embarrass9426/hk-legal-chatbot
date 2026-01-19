from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import os

model_id = "IEITYuan/Yuan-embedding-2.0-en"
save_dir = "backend/models/yuan-onnx-trt"
os.makedirs(save_dir, exist_ok=True)

# 💡 Make sure TensorRT DLLs are visible
venv = os.environ.get("VIRTUAL_ENV", "")
trt_libs = os.path.join(venv, "Lib", "site-packages", "tensorrt_libs")
os.environ["PATH"] = f"{trt_libs};{os.environ['PATH']}"

print("🚀 Exporting ONNX model for TensorRT...")
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    export=True,
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ],
    trust_remote_code=True,
    dtype="float16",          # FP16 → fastest for TensorRT
    opset=18,                # Opset 18 is widely supported
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ TensorRT-ready ONNX model saved to {save_dir}")