import os
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "IEITYuan/Yuan-embedding-2.0-en"
save_dir = "backend/models/yuan-onnx-trt"
os.makedirs(save_dir, exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "90"

# Export ONNX model for TensorRT
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    export=True,
    provider="TensorrtExecutionProvider",
    trust_remote_code=True,
    dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ ONNX model saved to {save_dir}")
