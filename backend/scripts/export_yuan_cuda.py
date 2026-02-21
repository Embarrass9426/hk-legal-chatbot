from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import os

model_id = "IEITYuan/Yuan-embedding-2.0-en"
# Go up one level: scripts -> backend -> models
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "yuan-onnx-cuda")
os.makedirs(save_dir, exist_ok=True)

print("Exporting ONNX model (CUDA FP32)...")

model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    export=True,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"CUDA-ready ONNX model saved to {save_dir}")
