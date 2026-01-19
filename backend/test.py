import os
venv = os.environ.get("VIRTUAL_ENV", "")
trt_libs = os.path.join(venv, "Lib", "site-packages", "tensorrt_libs")
if os.path.isdir(trt_libs):
    os.environ["PATH"] = f"{trt_libs};{os.environ['PATH']}"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(trt_libs)
    print("✅ TensorRT DLL path added:", trt_libs)
else:
    print("⚠️ TensorRT libs folder not found:", trt_libs)

import torch
import onnx
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort

# ==========================================================
# 1️⃣  Confirm that the ONNX file really has weights
# ==========================================================
path = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\yuan-onnx-trt\model.onnx"
m = onnx.load(path)
weights = [t.raw_data for t in m.graph.initializer]
total_bytes = sum(len(w) for w in weights)
print("ONNX initializers:", len(weights), "Total bytes:", total_bytes)
# → Should be a few GB → valid model

# ==========================================================
# 2️⃣  Load tokenizer and model
# ==========================================================
model_dir = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\yuan-onnx-trt"

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    fix_mistral_regex=True,
)

sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1

model = ORTModelForFeatureExtraction.from_pretrained(
    model_dir,
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    trust_remote_code=True,
    session_options=sess_opt,
)

print("✅ Providers:", model.model.get_providers())

# ==========================================================
# 3️⃣  Prepare inputs and force INT64 dtype
# ==========================================================
text = "This is a test section of a legal document."
inputs = tokenizer(text, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]
inputs["position_ids"] = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
for k in ["input_ids", "attention_mask", "position_ids"]:
    inputs[k] = inputs[k].to(torch.int64)  # make sure bindings match ONNX graph

# ==========================================================
# 4️⃣  Run inference and print all outputs for debugging
# ==========================================================
with torch.no_grad():
    outputs = model(**inputs)

print("Output keys:", outputs.keys() if isinstance(outputs, dict) else type(outputs))

if isinstance(outputs, dict):
    for k, v in outputs.items():
        print(k, v.shape, float(torch.abs(v).sum()))
else:
    print("Raw tensor output shape:", outputs[0].shape if isinstance(outputs, (list, tuple)) else outputs.shape)
    print("Sum of abs:", float(torch.abs(outputs[0]).sum() if isinstance(outputs, (list, tuple)) else torch.abs(outputs).sum()))