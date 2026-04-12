import os
import sys
import importlib
from typing import Any, Dict, List, Tuple

import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


model_id = "BAAI/bge-reranker-v2-m3"
save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "bge-reranker-v2-m3")
os.makedirs(save_dir, exist_ok=True)


def _collect_model_bytes(main_onnx_path: str) -> int:
    total = 0
    seen_files = set()

    base_name = os.path.basename(main_onnx_path)
    base_root, _ = os.path.splitext(base_name)
    parent = os.path.dirname(main_onnx_path)

    for name in os.listdir(parent):
        if name in {base_name, f"{base_root}.onnx_data", f"{base_root}.data"}:
            path = os.path.join(parent, name)
            if path not in seen_files:
                total += os.path.getsize(path)
                seen_files.add(path)
        elif name.startswith(base_root) and (
            name.endswith(".onnx_data") or name.endswith(".data")
        ):
            path = os.path.join(parent, name)
            if path not in seen_files:
                total += os.path.getsize(path)
                seen_files.add(path)
    return total


def _is_valid_converted_model(model: Any) -> bool:
    if model is None:
        return False
    if len(model.graph.node) == 0:
        return False
    if len(model.graph.initializer) == 0:
        return False
    return True


def _ort_loadable(model_path: str) -> Tuple[bool, str]:
    try:
        _ = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return True, ""
    except Exception as exc:
        return False, str(exc)


print(f"[INFO] Starting ONNX export for {model_id}...")
print(f"[INFO] Output directory: {save_dir}")

occ_converter = None
ort_converter = None

try:
    occ_float16 = importlib.import_module("onnxconverter_common.float16")
    occ_converter = occ_float16.convert_float_to_float16
except ImportError:
    pass

try:
    ort_float16 = importlib.import_module("onnxruntime.transformers.float16")
    ort_converter = ort_float16.convert_float_to_float16
except ImportError:
    pass

available_backends = []
if occ_converter is not None:
    available_backends.append("onnxconverter_common")
if ort_converter is not None:
    available_backends.append("onnxruntime.transformers.float16")

if not available_backends:
    print(
        "[ERROR] No FP16 converter backend available. Install either "
        "onnxconverter-common or onnxruntime."
    )
    sys.exit(1)

print(f"[INFO] FP16 converter backends: {', '.join(available_backends)}")


print("Step 1: Exporting base ONNX model (FP32) via Optimum...")
model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True,
    provider="CPUExecutionProvider",
)

model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)

onnx_path = os.path.join(save_dir, "model.onnx")
print(f"Base FP32 model saved to {onnx_path}")

print("Step 2: Converting to FP16 with ORT-loadability checks...")
try:
    conversion_attempts: List[Tuple[str, Any, Dict[str, Any]]] = []

    if occ_converter is not None:
        conversion_attempts.extend(
            [
                (
                    "onnxconverter_common",
                    occ_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": False,
                        "disable_shape_infer": True,
                        "force_fp16_initializers": True,
                    },
                ),
                (
                    "onnxconverter_common",
                    occ_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": True,
                        "disable_shape_infer": False,
                        "force_fp16_initializers": True,
                    },
                ),
                (
                    "onnxconverter_common",
                    occ_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": True,
                        "disable_shape_infer": False,
                        "force_fp16_initializers": True,
                        "op_block_list": [
                            "ReduceMean",
                            "Pow",
                            "Sqrt",
                            "LayerNormalization",
                        ],
                    },
                ),
            ]
        )

    if ort_converter is not None:
        conversion_attempts.extend(
            [
                (
                    "onnxruntime.transformers.float16",
                    ort_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": False,
                        "disable_shape_infer": True,
                        "force_fp16_initializers": True,
                    },
                ),
                (
                    "onnxruntime.transformers.float16",
                    ort_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": True,
                        "disable_shape_infer": False,
                        "force_fp16_initializers": True,
                    },
                ),
                (
                    "onnxruntime.transformers.float16",
                    ort_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": True,
                        "disable_shape_infer": False,
                        "force_fp16_initializers": True,
                        "op_block_list": [
                            "ReduceMean",
                            "Pow",
                            "Sqrt",
                            "LayerNormalization",
                        ],
                    },
                ),
            ]
        )

    model_fp16 = None
    last_error = None
    fp16_path = os.path.join(save_dir, "model_fp16.onnx")

    for attempt_idx, (backend_name, converter, kwargs) in enumerate(
        conversion_attempts, start=1
    ):
        print(
            f"[INFO] FP16 attempt {attempt_idx}/{len(conversion_attempts)} "
            f"via {backend_name} with kwargs={kwargs}"
        )

        model_fp32 = onnx.load(onnx_path, load_external_data=True)
        try:
            converted = converter(model_fp32, **kwargs)
            candidate = converted if converted is not None else model_fp32
            if not _is_valid_converted_model(candidate):
                last_error = RuntimeError(
                    f"Converted model invalid: nodes={len(candidate.graph.node)}, "
                    f"initializers={len(candidate.graph.initializer)}"
                )
                print(f"[WARN] FP16 attempt failed: {last_error}")
                continue

            onnx.save_model(
                candidate,
                fp16_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model_fp16.onnx_data",
                size_threshold=1024,
                convert_attribute=False,
            )

            load_ok, load_error = _ort_loadable(fp16_path)
            if load_ok:
                model_fp16 = candidate
                print(f"[INFO] FP16 conversion + ORT load succeeded via {backend_name}")
                break

            last_error = RuntimeError(load_error)
            print(f"[WARN] FP16 attempt produced non-loadable model: {load_error}")
        except Exception as attempt_exc:
            last_error = attempt_exc
            print(f"[WARN] FP16 attempt failed with exception: {attempt_exc}")

    if model_fp16 is None:
        print(
            "[ERROR] All FP16 conversion attempts failed ORT loadability checks. "
            "Refusing to emit a broken model_fp16.onnx."
        )
        if last_error is not None:
            print(f"[ERROR] Last failure: {last_error}")
        sys.exit(1)

    print(f"FP16 model saved to {fp16_path}")
    fp16_total_size_mb = _collect_model_bytes(fp16_path) / (1024 * 1024)
    print(f"FP16 model total size (onnx + external data): {fp16_total_size_mb:.1f} MB")

    if not os.path.exists(os.path.join(save_dir, "model_fp16.onnx_data")):
        print(
            "[WARN] model_fp16.onnx_data not found. "
            "If total size looks too small, conversion likely failed."
        )

    if fp16_total_size_mb < 100.0:
        print("[ERROR] FP16 export resulted in suspiciously small file! Aborting.")
        sys.exit(1)

except Exception as e:
    print(f"[ERROR] FP16 conversion failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("Step 3: Validating ONNX graph...")
try:
    onnx.checker.check_model(fp16_path)
    print("ONNX validation passed.")
except Exception as e:
    print(f"[WARN] ONNX validation issue: {e}")

load_ok, load_error = _ort_loadable(fp16_path)
if not load_ok:
    print(f"[ERROR] Final FP16 model failed ORT load check: {load_error}")
    sys.exit(1)

print()
print("Export complete!")
print(f"  FP32 model: {onnx_path}")
print(f"  FP16 model: {fp16_path}")
print()
print("Use strict reranker mode:")
print("  RERANKER_MODEL_DIR=bge-reranker-v2-m3")
print("  RERANKER_ONNX_FILE=model_fp16.onnx")
print("  RERANKER_STRICT_ONNX_FILE=1")
