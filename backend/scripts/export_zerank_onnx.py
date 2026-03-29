import os
import traceback

import onnx
from onnxconverter_common import float16
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


model_id = "zeroentropy/zerank-2"
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "zerank-onnx-trt")
os.makedirs(save_dir, exist_ok=True)


def main() -> None:
    print("[INFO] Starting export for zeroentropy/zerank-2...")

    try:
        print("[INFO] Step 1: Exporting ONNX FP32...")
        model = ORTModelForSequenceClassification.from_pretrained(
            model_id,
            export=True,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[INFO] Saved FP32 export to {save_dir}")

        print("[INFO] Step 2: Converting ONNX to FP16...")
        onnx_path = os.path.join(save_dir, "model.onnx")
        model_fp32 = onnx.load(onnx_path)
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            keep_io_types=False,
            op_block_list=["ReduceMean", "Pow", "Sqrt"],
        )

        fp16_path = os.path.join(save_dir, "model_fp16.onnx")
        onnx.save(model_fp16, fp16_path)

        fp16_size = os.path.getsize(fp16_path)
        print(f"[INFO] FP16 file size: {fp16_size / (1024 * 1024):.2f} MB")

        if fp16_size > 1 * 1024 * 1024:
            os.replace(fp16_path, onnx_path)
            print("[INFO] Replaced model.onnx with FP16 model.")
        else:
            print("[WARN] FP16 file is too small; keeping FP32 model.onnx.")
            if os.path.exists(fp16_path):
                os.remove(fp16_path)

        print("[INFO] Zerank export completed successfully.")

    except Exception as exc:
        print(f"[ERROR] Zerank export failed: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
