import importlib
import inspect
import os
import re
from typing import Any, Dict, List, Tuple

import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from transformers import AutoTokenizer


MODEL_ID = "zeroentropy/zerank-1-small"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "zerank-1-small")
os.makedirs(SAVE_DIR, exist_ok=True)


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


def _filter_kwargs_for_converter(
    converter: Any,
    desired_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    try:
        sig = inspect.signature(converter)
    except (TypeError, ValueError):
        return desired_kwargs, []

    params = sig.parameters
    supports_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if supports_var_kwargs:
        return desired_kwargs, []

    filtered = {k: v for k, v in desired_kwargs.items() if k in params}
    ignored = sorted(set(desired_kwargs.keys()) - set(filtered.keys()))
    return filtered, ignored


def _count_fp16_initializers(model: Any) -> int:
    float16_tensor_type = int(getattr(onnx.TensorProto, "FLOAT16"))
    count = 0
    for initializer in model.graph.initializer:
        if int(initializer.data_type) == float16_tensor_type:
            count += 1
    return count


def _collect_tensor_elem_types(model: Any) -> Dict[str, int]:
    tensor_types: Dict[str, int] = {}

    def _record_vi(vi: Any) -> None:
        if not vi.name:
            return
        tensor_type = vi.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            return
        elem_type = int(tensor_type.elem_type)
        if elem_type != 0:
            tensor_types[vi.name] = elem_type

    for initializer in model.graph.initializer:
        tensor_types[initializer.name] = int(initializer.data_type)

    for vi in model.graph.input:
        _record_vi(vi)
    for vi in model.graph.value_info:
        _record_vi(vi)
    for vi in model.graph.output:
        _record_vi(vi)

    return tensor_types


def _convert_initializer_dtype(initializer: Any, target_dtype: int) -> Any:
    np_dtype = None
    if target_dtype == int(onnx.TensorProto.FLOAT16):
        import numpy as np

        np_dtype = np.float16
    elif target_dtype == int(onnx.TensorProto.FLOAT):
        import numpy as np

        np_dtype = np.float32
    else:
        return initializer

    array = numpy_helper.to_array(initializer)
    if array.dtype == np_dtype:
        return initializer
    converted = array.astype(np_dtype)
    return numpy_helper.from_array(converted, name=initializer.name)


def _harmonize_mixed_binary_inputs(model: Any) -> Tuple[Any, int]:
    float16_type = int(onnx.TensorProto.FLOAT16)
    float32_type = int(onnx.TensorProto.FLOAT)
    allowed = {float16_type, float32_type}

    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
        working_model = inferred_model
    except Exception:
        working_model = model

    tensor_types = _collect_tensor_elem_types(working_model)

    initializer_map = {init.name: init for init in model.graph.initializer}
    initializer_indices = {
        init.name: idx for idx, init in enumerate(model.graph.initializer)
    }

    new_nodes = []
    fixes = 0
    cast_counter = 0

    binary_ops = {"Add", "Sub", "Mul", "Div"}

    for node in model.graph.node:
        if node.op_type not in binary_ops or len(node.input) < 2:
            new_nodes.append(node)
            continue

        in0 = node.input[0]
        in1 = node.input[1]
        t0 = tensor_types.get(in0)
        t1 = tensor_types.get(in1)

        if (
            t0 is None
            or t1 is None
            or t0 == t1
            or t0 not in allowed
            or t1 not in allowed
        ):
            new_nodes.append(node)
            continue

        out_type = None
        if node.output:
            out_type = tensor_types.get(node.output[0])

        if out_type in allowed:
            target_type = out_type
        elif in0 in initializer_map and in1 not in initializer_map:
            target_type = t1
        elif in1 in initializer_map and in0 not in initializer_map:
            target_type = t0
        elif t0 == float16_type or t1 == float16_type:
            target_type = float16_type
        else:
            target_type = t0

        replacement_inputs = [in0, in1]
        for idx, (inp_name, inp_type) in enumerate([(in0, t0), (in1, t1)]):
            if inp_type == target_type:
                continue

            if inp_name in initializer_map:
                init_idx = initializer_indices[inp_name]
                model.graph.initializer[init_idx].CopyFrom(
                    _convert_initializer_dtype(
                        model.graph.initializer[init_idx], target_type
                    )
                )
                tensor_types[inp_name] = target_type
                continue

            cast_name = (
                f"{node.name or node.output[0] if node.output else node.op_type.lower()}"
                f"_cast_{cast_counter}"
            )
            cast_output = f"{inp_name}_cast_to_{target_type}_{cast_counter}"
            cast_counter += 1
            cast_node = helper.make_node(
                "Cast",
                inputs=[inp_name],
                outputs=[cast_output],
                name=cast_name,
                to=target_type,
            )
            new_nodes.append(cast_node)
            replacement_inputs[idx] = cast_output
            tensor_types[cast_output] = target_type

        fixed_node = helper.make_node(
            node.op_type,
            inputs=replacement_inputs,
            outputs=list(node.output),
            name=node.name,
        )
        for attr in node.attribute:
            fixed_node.attribute.extend([attr])
        new_nodes.append(fixed_node)
        fixes += 1

    if fixes == 0:
        return model, 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, fixes


def _iterative_binary_harmonization(model: Any, max_rounds: int = 4) -> Tuple[Any, int]:
    total_fixes = 0
    working = model
    for _ in range(max_rounds):
        working, fixes = _harmonize_mixed_binary_inputs(working)
        total_fixes += fixes
        if fixes == 0:
            break
    return working, total_fixes


def _cleanup_fp16_outputs(fp16_path: str) -> None:
    parent = os.path.dirname(fp16_path)
    base_name = os.path.basename(fp16_path)
    base_root, _ = os.path.splitext(base_name)
    candidates = [
        fp16_path,
        os.path.join(parent, f"{base_root}.onnx_data"),
        os.path.join(parent, f"{base_root}.data"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _build_fp16_converters() -> List[Tuple[str, Any, Dict[str, Any]]]:
    converters: List[Tuple[str, Any, Dict[str, Any]]] = []
    allow_experimental_ort = os.getenv(
        "ZERANK_EXPORT_ALLOW_EXPERIMENTAL_ORT_FP16", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        occ_float16 = importlib.import_module("onnxconverter_common.float16")
        occ_converter = occ_float16.convert_float_to_float16
        converters.extend(
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
                        "op_block_list": [
                            "LayerNormalization",
                            "SimplifiedLayerNormalization",
                            "ReduceMean",
                            "Pow",
                            "Sqrt",
                        ],
                    },
                ),
                (
                    "onnxconverter_common",
                    occ_converter,
                    {
                        "min_positive_val": 5.96e-08,
                        "max_finite_val": 65504.0,
                        "keep_io_types": False,
                        "disable_shape_infer": False,
                        "force_fp16_initializers": True,
                        "node_block_list": [
                            "/model/layers.0/input_layernorm/Add",
                            "/model/layers.0/input_layernorm/Div",
                        ],
                        "op_block_list": [
                            "Add",
                            "Sub",
                            "Mul",
                            "Div",
                            "LayerNormalization",
                            "SimplifiedLayerNormalization",
                            "ReduceMean",
                            "Pow",
                            "Sqrt",
                        ],
                    },
                ),
            ]
        )
    except ImportError:
        pass

    if allow_experimental_ort:
        try:
            ort_float16 = importlib.import_module("onnxruntime.transformers.float16")
            ort_converter = ort_float16.convert_float_to_float16
            converters.extend(
                [
                    (
                        "onnxruntime.transformers.float16",
                        ort_converter,
                        {
                            "min_positive_val": 5.96e-08,
                            "max_finite_val": 65504.0,
                            "keep_io_types": True,
                            "disable_shape_infer": True,
                            "force_fp16_initializers": True,
                        },
                    ),
                ]
            )
        except ImportError:
            pass
    else:
        print(
            "[INFO] Skipping experimental onnxruntime.transformers.float16 "
            "converter. Set ZERANK_EXPORT_ALLOW_EXPERIMENTAL_ORT_FP16=1 to enable."
        )

    return converters


def _export_base_onnx(onnx_path: str) -> None:
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            MODEL_ID,
            trust_remote_code=True,
            backend="onnx",
            model_kwargs={
                "export": True,
                "provider": "CPUExecutionProvider",
                "file_name": "model.onnx",
            },
        )
        model.save_pretrained(SAVE_DIR)
    except Exception as st_exc:
        print(f"[WARN] sentence-transformers export failed: {st_exc}")
        print("[INFO] Falling back to Optimum ORT export")

        from optimum.onnxruntime import ORTModelForSequenceClassification

        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            export=True,
            provider="CPUExecutionProvider",
            trust_remote_code=True,
        )
        model.save_pretrained(SAVE_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(SAVE_DIR)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Base ONNX export failed, file not found: {onnx_path}")


def main() -> int:
    print(f"[INFO] Starting ONNX export for {MODEL_ID}")
    print(f"[INFO] Output directory: {SAVE_DIR}")

    onnx_path = os.path.join(SAVE_DIR, "model.onnx")
    fp16_path = os.path.join(SAVE_DIR, "model_fp16.onnx")

    skip_export_if_present = os.getenv(
        "ZERANK_EXPORT_SKIP_EXPORT_IF_PRESENT", "1"
    ).strip().lower() in {"1", "true", "yes"}

    if skip_export_if_present and os.path.exists(onnx_path):
        print("[INFO] Reusing existing FP32 ONNX model (skip export enabled)")
    else:
        try:
            _export_base_onnx(onnx_path)
            print(f"[INFO] Base FP32 model saved: {onnx_path}")
        except Exception as exc:
            if os.path.exists(onnx_path):
                print(
                    "[WARN] Base export failed, but existing model.onnx found. "
                    f"Continuing with existing file. Reason: {exc}"
                )
            else:
                raise

    fp32_ok, fp32_err = _ort_loadable(onnx_path)
    if not fp32_ok:
        print(f"[ERROR] Base FP32 model is not loadable by ORT: {fp32_err}")
        return 1

    converters = _build_fp16_converters()
    if not converters:
        print(
            "[ERROR] No FP16 converter backend available. Install onnxconverter-common or use onnxruntime transformers tools."
        )
        return 1

    model_fp16 = None
    last_error: Exception | None = None
    for attempt_idx, (backend_name, converter, desired_kwargs) in enumerate(
        converters, start=1
    ):
        kwargs, ignored_kwargs = _filter_kwargs_for_converter(converter, desired_kwargs)
        print(
            f"[INFO] FP16 attempt {attempt_idx}/{len(converters)} via {backend_name} kwargs={kwargs}"
        )
        if ignored_kwargs:
            print(
                f"[INFO] Ignored unsupported kwargs for {backend_name}: {ignored_kwargs}"
            )

        _cleanup_fp16_outputs(fp16_path)
        model_fp32 = onnx.load(onnx_path, load_external_data=True)
        try:
            converted = converter(model_fp32, **kwargs)
            candidate = converted if converted is not None else model_fp32
            if not _is_valid_converted_model(candidate):
                raise RuntimeError("Converted model is empty/invalid")

            fp16_initializer_count = _count_fp16_initializers(candidate)
            if fp16_initializer_count == 0:
                raise RuntimeError("No FLOAT16 initializers found after conversion")

            onnx.save_model(
                candidate,
                fp16_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model_fp16.onnx_data",
                size_threshold=1024,
                convert_attribute=False,
            )

            ok, err = _ort_loadable(fp16_path)
            if ok:
                model_fp16 = candidate
                print(f"[INFO] FP16 conversion + ORT load succeeded via {backend_name}")
                print(f"[INFO] FLOAT16 initializer count: {fp16_initializer_count}")
                break

            if re.search(
                r"Type parameter \(T\) of Optype \((Add|Sub|Mul|Div)\) bound to different types",
                err,
            ):
                print(
                    "[WARN] Mixed-type binary op detected; attempting in-graph harmonization pass"
                )
                fixed_candidate, fix_count = _iterative_binary_harmonization(candidate)
                if fix_count == 0:
                    raise RuntimeError(
                        "Mixed float/float16 binary node types detected after conversion: "
                        + err
                    )

                onnx.save_model(
                    fixed_candidate,
                    fp16_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="model_fp16.onnx_data",
                    size_threshold=1024,
                    convert_attribute=False,
                )
                fixed_ok, fixed_err = _ort_loadable(fp16_path)
                if fixed_ok:
                    model_fp16 = fixed_candidate
                    print(
                        "[INFO] Harmonization pass fixed mixed binary-op inputs "
                        f"(nodes fixed: {fix_count})"
                    )
                    break

                if re.search(
                    r"Type parameter \(T\) of Optype \((Add|Sub|Mul|Div)\) bound to different types",
                    fixed_err,
                ):
                    second_candidate, second_fix_count = (
                        _iterative_binary_harmonization(fixed_candidate)
                    )
                    if second_fix_count > 0:
                        onnx.save_model(
                            second_candidate,
                            fp16_path,
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="model_fp16.onnx_data",
                            size_threshold=1024,
                            convert_attribute=False,
                        )
                        second_ok, second_err = _ort_loadable(fp16_path)
                        if second_ok:
                            model_fp16 = second_candidate
                            print(
                                "[INFO] Second harmonization pass succeeded "
                                f"(additional nodes fixed: {second_fix_count})"
                            )
                            break
                        fixed_err = second_err

                raise RuntimeError(
                    "Mixed Add harmonization did not produce ORT-loadable model: "
                    + fixed_err
                )

            if backend_name == "onnxruntime.transformers.float16" and re.search(
                r"SimplifiedLayerNormFusion|GetIndexFromName|InsertedPrecisionFreeCast",
                err,
            ):
                raise RuntimeError(
                    "Known ORT fp16 cast/fusion incompatibility detected: " + err
                )
            raise RuntimeError(err)
        except Exception as exc:
            last_error = exc
            print(f"[WARN] FP16 attempt failed: {exc}")

    if model_fp16 is None:
        require_fp16 = os.getenv("ZERANK_EXPORT_REQUIRE_FP16", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        print("[ERROR] All FP16 conversion attempts failed.")
        if last_error is not None:
            print(f"[ERROR] Last failure: {last_error}")
        if require_fp16:
            return 1
        print(
            "[WARN] Proceeding with FP32-only export. "
            "Set ZERANK_EXPORT_REQUIRE_FP16=1 to hard-fail when FP16 is unavailable."
        )
        return 0

    try:
        onnx.checker.check_model(fp16_path)
    except Exception as exc:
        print(f"[WARN] ONNX checker warning: {exc}")

    ok, err = _ort_loadable(fp16_path)
    if not ok:
        print(f"[ERROR] Final FP16 model not loadable by ORT: {err}")
        return 1

    fp16_mb = _collect_model_bytes(fp16_path) / (1024 * 1024)
    print(f"[INFO] FP16 model total size (onnx + external data): {fp16_mb:.1f} MB")
    print("[INFO] Export complete")
    print(f"  FP32 model: {onnx_path}")
    print(f"  FP16 model: {fp16_path}")
    print("[INFO] Use in eval variants as: zerank-1-small:model_fp16.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
