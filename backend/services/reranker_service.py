import json
import math
import os
import sys
import gc
import re
import threading
import unicodedata
from typing import Any, List, Optional, Tuple

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env

setup_env.setup_cuda_dlls()

import torch
import numpy as np
import onnxruntime as ort
import transformers
from langchain_core.documents import Document
from transformers import AutoTokenizer


class RerankerService:
    _instance = None
    _init_lock = threading.Lock()
    _bootstrap_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    print("[RerankerService] Creating NEW singleton instance")
                    cls._instance = super(RerankerService, cls).__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._rerank_lock = threading.Lock()
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_rerank_lock"):
            self._rerank_lock = threading.Lock()

        if self._initialized:
            return

        with self._bootstrap_lock:
            if self._initialized:
                return

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir_name = os.getenv("RERANKER_MODEL_DIR", "bge-reranker-v2-m3")
            self.model_path = os.path.join(base_dir, "models", model_dir_name)
            self.onnx_model_file = os.getenv("RERANKER_ONNX_FILE", "model_fp16.onnx")
            self.strict_onnx_file = os.getenv(
                "RERANKER_STRICT_ONNX_FILE", "1"
            ).strip().lower() in {"1", "true", "yes"}
            self.strict_mistral_regex_fix = os.getenv(
                "RERANKER_STRICT_MISTRAL_REGEX_FIX", "0"
            ).strip().lower() in {"1", "true", "yes"}
            self.tokenizer = None
            self.session = None
            self._model_loaded = False
            self._active_onnx_path = ""
            self._trt_runtime_disabled = False
            self._cuda_runtime_only = False
            self._trt_demoted_to_cuda = False
            self.trt_engine_cache_enable = True
            self.trt_engine_cache_path = os.path.join(self.model_path, "cache")
            self.trt_max_workspace_size = int(
                os.getenv("RERANKER_TRT_MAX_WORKSPACE_SIZE", str(268435456))
            )
            self.trt_min_subgraph_size = int(
                os.getenv("RERANKER_TRT_MIN_SUBGRAPH_SIZE", "8")
            )
            self.trt_max_partition_iterations = int(
                os.getenv("RERANKER_TRT_MAX_PARTITION_ITERATIONS", "200")
            )
            self.trt_builder_optimization_level = int(
                os.getenv("RERANKER_TRT_BUILDER_OPT_LEVEL", "2")
            )
            self.trt_context_memory_sharing_enable = os.getenv(
                "RERANKER_TRT_CONTEXT_MEMORY_SHARING_ENABLE", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.trt_op_types_to_exclude = os.getenv(
                "RERANKER_TRT_OP_TYPES_TO_EXCLUDE", "Range,Gather"
            ).strip()
            self.trt_profile_min_shapes = os.getenv(
                "RERANKER_TRT_PROFILE_MIN_SHAPES", ""
            ).strip()
            self.trt_profile_opt_shapes = os.getenv(
                "RERANKER_TRT_PROFILE_OPT_SHAPES", ""
            ).strip()
            self.trt_profile_max_shapes = os.getenv(
                "RERANKER_TRT_PROFILE_MAX_SHAPES", ""
            ).strip()
            self.rerank_micro_batch_size = int(
                os.getenv("RERANKER_MICRO_BATCH_SIZE", "8")
            )
            self.rerank_max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
            self.fixed_batch_size = int(os.getenv("RERANKER_FIXED_BATCH_SIZE", "1"))
            self.fixed_shape_padding = os.getenv(
                "RERANKER_FIXED_SHAPE_PADDING", "1"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.disable_runtime_cpu_fallback = os.getenv(
                "RERANKER_DISABLE_RUNTIME_CPU_FALLBACK", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.disable_ort_cpu_ep_fallback = os.getenv(
                "RERANKER_DISABLE_ORT_CPU_EP_FALLBACK", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.require_tensorrt = os.getenv(
                "RERANKER_REQUIRE_TENSORRT", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.force_cpu = os.getenv("RERANKER_FORCE_CPU", "0").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.enable_trt_experimental = os.getenv(
                "RERANKER_ENABLE_TRT_EXPERIMENTAL", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            self.ort_log_severity_level = int(
                os.getenv("RERANKER_ORT_LOG_SEVERITY_LEVEL", "2")
            )

            self._is_causal_lm = False
            self._yes_token_id: Optional[int] = None

            self._initialized = True

    def ensure_loaded(self):
        if self._model_loaded:
            return

        with self._bootstrap_lock:
            if self._model_loaded:
                return
            self._load_model()
            self._model_loaded = True

    def is_loaded(self) -> bool:
        return self._model_loaded

    def unload(self):
        with self._bootstrap_lock:
            self.session = None
            self.tokenizer = None
            self._active_onnx_path = ""
            self._cuda_runtime_only = False
            self._trt_runtime_disabled = False
            self._trt_demoted_to_cuda = False
            self._is_causal_lm = False
            self._yes_token_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model_loaded = False
            print("[RerankerService] Model unloaded.")

    def _build_provider_candidates(self) -> List[List[Any]]:
        available_providers = ort.get_available_providers()
        has_tensorrt_provider = "TensorrtExecutionProvider" in available_providers
        has_cuda_provider = "CUDAExecutionProvider" in available_providers

        # --- Demoted paths (one-way latches set at runtime) ---

        if self._trt_demoted_to_cuda:
            if has_cuda_provider:
                print(
                    "[RerankerService] TRT demoted to CUDA+CPU for this process lifetime."
                )
                return [["CUDAExecutionProvider", "CPUExecutionProvider"]]
            print("[RerankerService] TRT demoted but CUDA unavailable; using CPU only.")
            return [["CPUExecutionProvider"]]

        if self._cuda_runtime_only:
            if has_cuda_provider:
                print(
                    "[RerankerService] Using CUDAExecutionProvider only "
                    "(runtime TRT engine build failure)."
                )
                return [["CUDAExecutionProvider"]]
            print(
                "[RerankerService] CUDA runtime-only fallback requested but CUDA provider "
                "is unavailable. Falling back to CPUExecutionProvider."
            )
            return [["CPUExecutionProvider"]]

        if self.force_cpu or self._trt_runtime_disabled:
            reason = "RERANKER_FORCE_CPU=1" if self.force_cpu else "runtime TRT failure"
            print(f"[RerankerService] Using CPUExecutionProvider only ({reason}).")
            return [["CPUExecutionProvider"]]

        # --- Build TRT options (shared between strict-TRT and experimental) ---

        trt_provider_options = None
        if has_tensorrt_provider:
            trt_provider_options = {
                "trt_engine_cache_enable": self.trt_engine_cache_enable,
                "trt_engine_cache_path": self.trt_engine_cache_path,
                "trt_max_workspace_size": self.trt_max_workspace_size,
                "trt_min_subgraph_size": self.trt_min_subgraph_size,
                "trt_max_partition_iterations": self.trt_max_partition_iterations,
                "trt_builder_optimization_level": self.trt_builder_optimization_level,
                "trt_context_memory_sharing_enable": self.trt_context_memory_sharing_enable,
                "trt_force_sequential_engine_build": True,
            }

            if self.trt_op_types_to_exclude:
                trt_provider_options["trt_op_types_to_exclude"] = (
                    self.trt_op_types_to_exclude
                )

            if self.trt_profile_min_shapes:
                trt_provider_options["trt_profile_min_shapes"] = (
                    self.trt_profile_min_shapes
                )
            if self.trt_profile_opt_shapes:
                trt_provider_options["trt_profile_opt_shapes"] = (
                    self.trt_profile_opt_shapes
                )
            if self.trt_profile_max_shapes:
                trt_provider_options["trt_profile_max_shapes"] = (
                    self.trt_profile_max_shapes
                )

        # --- Tier construction ---

        tier_candidates: List[List[Any]] = []

        # Tier A: Experimental TRT → CUDA → CPU (3-provider chain)
        if (
            self.enable_trt_experimental
            and has_tensorrt_provider
            and has_cuda_provider
            and trt_provider_options is not None
        ):
            experimental_trt_opts = dict(trt_provider_options)
            # Safety: raise min subgraph size so TRT only claims large subgraphs
            experimental_trt_opts["trt_min_subgraph_size"] = max(
                20, experimental_trt_opts.get("trt_min_subgraph_size", 8)
            )
            tier_candidates.append(
                [
                    ("TensorrtExecutionProvider", experimental_trt_opts),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            )
            print(
                "[RerankerService] Experimental TRT→CUDA→CPU tier enabled "
                f"(trt_min_subgraph_size={experimental_trt_opts['trt_min_subgraph_size']})."
            )

        # Tier B: Strict TRT-only candidates
        if has_tensorrt_provider and trt_provider_options is not None:
            profile_retry_options = dict(trt_provider_options)
            profile_retry_options.pop("trt_profile_min_shapes", None)
            profile_retry_options.pop("trt_profile_opt_shapes", None)
            profile_retry_options.pop("trt_profile_max_shapes", None)

            trt_candidates: List[Any] = [
                (
                    "TensorrtExecutionProvider",
                    trt_provider_options,
                )
            ]

            op_exclude_fallback_options = dict(profile_retry_options)
            raw_excludes = [
                token.strip()
                for token in self.trt_op_types_to_exclude.split(",")
                if token.strip()
            ]
            if (
                len(raw_excludes) > 1
                and "trt_op_types_to_exclude" in op_exclude_fallback_options
            ):
                op_exclude_fallback_options["trt_op_types_to_exclude"] = raw_excludes[0]
                trt_candidates.append(
                    (
                        "TensorrtExecutionProvider",
                        op_exclude_fallback_options,
                    )
                )

            no_exclude_options = dict(profile_retry_options)
            no_exclude_options.pop("trt_op_types_to_exclude", None)
            if no_exclude_options != profile_retry_options:
                trt_candidates.append(
                    (
                        "TensorrtExecutionProvider",
                        no_exclude_options,
                    )
                )

            low_partition_options = dict(no_exclude_options)
            low_partition_options["trt_min_subgraph_size"] = 1
            low_partition_options["trt_max_partition_iterations"] = 10
            trt_candidates.append(
                (
                    "TensorrtExecutionProvider",
                    low_partition_options,
                )
            )

            if profile_retry_options != trt_provider_options:
                trt_candidates.append(
                    (
                        "TensorrtExecutionProvider",
                        profile_retry_options,
                    )
                )

            for trt_candidate in trt_candidates:
                tier_candidates.append([trt_candidate])

        if self.require_tensorrt and not has_tensorrt_provider:
            raise RuntimeError(
                "RERANKER_REQUIRE_TENSORRT=1 but TensorrtExecutionProvider is unavailable. "
                f"Available providers: {available_providers}"
            )

        # Tier C: CUDA + CPU (proven stable fallback — the immediate win)
        if has_cuda_provider:
            tier_candidates.append(["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Tier D: CPU only (last resort)
        tier_candidates.append(["CPUExecutionProvider"])

        if not has_tensorrt_provider and not self.require_tensorrt:
            print(
                "[RerankerService] TensorrtExecutionProvider unavailable. "
                f"Available providers: {available_providers}."
            )

        return tier_candidates

    @staticmethod
    def _is_unsafe_mixed_provider_runtime(active_providers: Any) -> bool:
        has_trt = "TensorrtExecutionProvider" in active_providers
        has_cpu = "CPUExecutionProvider" in active_providers
        has_cuda = "CUDAExecutionProvider" in active_providers
        return has_trt and has_cpu and not has_cuda

    def _create_session_with_fallback(self, sess_opt: ort.SessionOptions) -> None:
        preferred_onnx_path = os.path.join(self.model_path, self.onnx_model_file)
        fallback_onnx_path = os.path.join(self.model_path, "model.onnx")
        candidate_onnx_paths: List[str] = []

        if os.path.exists(preferred_onnx_path):
            candidate_onnx_paths.append(preferred_onnx_path)

        if (not self.strict_onnx_file) and os.path.exists(fallback_onnx_path):
            if fallback_onnx_path not in candidate_onnx_paths:
                candidate_onnx_paths.append(fallback_onnx_path)

        if not candidate_onnx_paths:
            if self.strict_onnx_file:
                raise FileNotFoundError(
                    "Strict ONNX mode enabled, but preferred model file not found: "
                    f"{preferred_onnx_path}"
                )
            raise FileNotFoundError(
                f"No ONNX model file found. Checked: {preferred_onnx_path}, {fallback_onnx_path}"
            )

        session_errors: List[str] = []
        self.session = None
        self._active_onnx_path = ""

        for providers in self._build_provider_candidates():
            for candidate_onnx_path in candidate_onnx_paths:
                try:
                    attempt_sess_opt = sess_opt
                    is_strict_trt_only = (
                        providers
                        and isinstance(providers[0], tuple)
                        and providers[0][0] == "TensorrtExecutionProvider"
                        and len(providers) == 1
                    )
                    is_trt_mixed = (
                        providers
                        and isinstance(providers[0], tuple)
                        and providers[0][0] == "TensorrtExecutionProvider"
                        and len(providers) > 1
                    )
                    if is_strict_trt_only:
                        attempt_sess_opt = ort.SessionOptions()
                        attempt_sess_opt.intra_op_num_threads = (
                            sess_opt.intra_op_num_threads
                        )
                        attempt_sess_opt.inter_op_num_threads = (
                            sess_opt.inter_op_num_threads
                        )
                        attempt_sess_opt.log_severity_level = (
                            sess_opt.log_severity_level
                        )
                        attempt_sess_opt.log_verbosity_level = (
                            sess_opt.log_verbosity_level
                        )
                        try:
                            attempt_sess_opt.add_session_config_entry(
                                "session.disable_cpu_ep_fallback", "1"
                            )
                        except Exception as config_exc:
                            print(
                                "[RerankerService] Could not enforce strict non-CPU session "
                                f"candidate; skipping provider chain {providers}. Error: {config_exc}"
                            )
                            session_errors.append(
                                f"providers={providers}, model={candidate_onnx_path}: "
                                f"strict session config rejected: {config_exc}"
                            )
                            continue
                    elif is_trt_mixed:
                        attempt_sess_opt = ort.SessionOptions()
                        attempt_sess_opt.intra_op_num_threads = (
                            sess_opt.intra_op_num_threads
                        )
                        attempt_sess_opt.inter_op_num_threads = (
                            sess_opt.inter_op_num_threads
                        )
                        attempt_sess_opt.log_severity_level = (
                            sess_opt.log_severity_level
                        )
                        attempt_sess_opt.log_verbosity_level = (
                            sess_opt.log_verbosity_level
                        )

                    self.session = ort.InferenceSession(
                        candidate_onnx_path,
                        sess_options=attempt_sess_opt,
                        providers=providers,
                        disabled_optimizers=None,
                    )
                    active_providers = self.session.get_providers()
                    if self._is_unsafe_mixed_provider_runtime(active_providers):
                        print(
                            "[RerankerService] Rejecting unsafe mixed provider runtime: "
                            f"{active_providers}. Trying next candidate."
                        )
                        session_errors.append(
                            f"providers={providers}, model={candidate_onnx_path}: "
                            f"unsafe mixed runtime providers={active_providers}"
                        )
                        self.session = None
                        continue
                    self._active_onnx_path = candidate_onnx_path
                    break
                except Exception as exc:
                    if (
                        providers
                        and isinstance(providers[0], tuple)
                        and "dynamic shape inputs with associated profiles" in str(exc)
                    ):
                        print(
                            "[RerankerService] Explicit TRT profiles rejected because ORT "
                            "requires profile coverage for dynamic internal subgraph inputs; "
                            "retrying TRT without explicit profile options."
                        )
                    if providers and isinstance(providers[0], tuple):
                        provider_name = providers[0][0]
                        provider_options = providers[0][1]
                        print(
                            "[RerankerService] Session init failure with provider "
                            f"{provider_name} and options={provider_options}: {exc}"
                        )
                    session_errors.append(
                        f"providers={providers}, model={candidate_onnx_path}: {exc}"
                    )
            if self.session is not None:
                break

        if self.session is None:
            raise RuntimeError(
                "Failed to create ONNX Runtime session for reranker. "
                + " | ".join(session_errors)
            )

        if self._active_onnx_path != preferred_onnx_path:
            if self.strict_onnx_file:
                raise RuntimeError(
                    "Strict ONNX mode enabled; fallback ONNX path was selected unexpectedly: "
                    f"{self._active_onnx_path}. Set RERANKER_STRICT_ONNX_FILE=0 to allow fallback."
                )
            print(
                "[RerankerService] Preferred ONNX model failed; using fallback model: "
                f"{self._active_onnx_path}"
            )

    def _load_model(self):
        print(f"[RerankerService] Loading model from: {self.model_path}")

        try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
                print("[RerankerService] Tokenizer loaded with fix_mistral_regex=True")
            except TypeError:
                if self.strict_mistral_regex_fix:
                    raise RuntimeError(
                        "Installed transformers does not support fix_mistral_regex for "
                        "AutoTokenizer.from_pretrained. Upgrade transformers in the active "
                        "runtime environment and retry. "
                        f"Current transformers={transformers.__version__}."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
                print(
                    "[RerankerService] fix_mistral_regex unsupported in this runtime; "
                    "continuing with tokenizer fallback. "
                    f"transformers={transformers.__version__}."
                )
        except Exception as e:
            print(f"[RerankerService] Failed to load tokenizer: {e}")
            raise

        self._detect_causal_lm_architecture()

        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1
        sess_opt.log_severity_level = self.ort_log_severity_level
        if self.ort_log_severity_level == 0:
            sess_opt.log_verbosity_level = 1
            print(
                "[RerankerService] ORT VERBOSE logging enabled "
                "(log_severity_level=0, log_verbosity_level=1). "
                "Node placement details will be printed to stderr."
            )

        os.makedirs(self.trt_engine_cache_path, exist_ok=True)

        self._create_session_with_fallback(sess_opt)

        if self.session is None:
            raise RuntimeError("Reranker session was not created.")

        active_providers = self.session.get_providers()
        print(f"[RerankerService] Active Providers: {active_providers}")
        print(
            "[RerankerService] Runtime config: "
            f"micro_batch={self.rerank_micro_batch_size}, max_length={self.rerank_max_length}, "
            f"fixed_batch_size={self.fixed_batch_size}, "
            f"fixed_shape_padding={self.fixed_shape_padding}, "
            f"trt_context_memory_sharing_enable={self.trt_context_memory_sharing_enable}, "
            f"disable_ort_cpu_ep_fallback={self.disable_ort_cpu_ep_fallback}, "
            f"ort_log_severity_level={self.ort_log_severity_level}"
        )

        if (
            self.require_tensorrt
            and "TensorrtExecutionProvider" not in active_providers
        ):
            raise RuntimeError(
                "RERANKER_REQUIRE_TENSORRT=1 but TensorRT is not active for reranker session. "
                f"Active providers: {active_providers}"
            )

        if "CUDAExecutionProvider" in active_providers:
            print(
                "[RerankerService] CUDAExecutionProvider is active for reranker fallback path."
            )

    def configure_model(self, model_dir_name: str, onnx_model_file: str) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        next_model_path = os.path.join(base_dir, "models", model_dir_name)
        next_onnx_file = onnx_model_file

        if (
            self.model_path == next_model_path
            and self.onnx_model_file == next_onnx_file
        ):
            return

        if self._model_loaded:
            self.unload()

        self.model_path = next_model_path
        self.onnx_model_file = next_onnx_file
        self._trt_runtime_disabled = False
        self._trt_demoted_to_cuda = False
        self.trt_engine_cache_path = os.path.join(self.model_path, "cache")
        print(
            "[RerankerService] Reconfigured model to "
            f"dir={model_dir_name}, onnx={onnx_model_file}"
        )

    @staticmethod
    def _is_tensorrt_runtime_failure(error_message: str) -> bool:
        lowered = error_message.lower()
        return (
            "tensorrt ep failed to create engine" in lowered
            or "non-zero status code returned while running trtkernel" in lowered
            or "ep_fail" in lowered
        )

    _CAUSAL_LM_ARCHITECTURES = frozenset(
        {
            "Qwen3ForCausalLM",
            "LlamaForCausalLM",
            "Gemma3ForCausalLM",
            "Gemma3ForConditionalGeneration",
        }
    )

    def _detect_causal_lm_architecture(self) -> None:
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.isfile(config_path):
            self._is_causal_lm = False
            self._yes_token_id = None
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[RerankerService] Could not read config.json: {exc}")
            self._is_causal_lm = False
            self._yes_token_id = None
            return

        architectures = config.get("architectures", [])
        detected = any(arch in self._CAUSAL_LM_ARCHITECTURES for arch in architectures)

        if not detected:
            self._is_causal_lm = False
            self._yes_token_id = None
            print(
                f"[RerankerService] Classification model detected "
                f"(architectures={architectures}). Using standard score extraction."
            )
            return

        if self.tokenizer is None:
            raise RuntimeError(
                "Causal LM architecture detected but tokenizer is not loaded."
            )

        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        if not yes_ids:
            raise RuntimeError(
                "Causal LM reranker requires a 'Yes' token but tokenizer returned "
                "empty encoding for 'Yes'."
            )

        self._is_causal_lm = True
        self._yes_token_id = yes_ids[0]
        print(
            f"[RerankerService] Causal LM reranker detected "
            f"(architectures={architectures}). "
            f"yes_token_id={self._yes_token_id}. "
            f"Score extraction: last-token Yes logit / 5.0 → sigmoid."
        )

    def _format_chat_pairs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[str]:
        assert self.tokenizer is not None
        formatted: List[str] = []
        for query, document in pairs:
            messages = [
                {"role": "system", "content": query},
                {"role": "user", "content": document},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted.append(str(text))
        return formatted

    def _extract_causal_lm_scores(
        self,
        logits: np.ndarray,
        ort_inputs: dict[str, Any],
        batch_size: int,
    ) -> np.ndarray:
        assert self._yes_token_id is not None

        if logits.ndim == 2 and logits.shape[1] == 2:
            yes_logits = logits[:, 0].astype(np.float64)
            scaled = yes_logits / 5.0
            return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)

        if logits.ndim == 2 and logits.shape[1] == 1:
            yes_logits = logits[:, 0].astype(np.float64)
            scaled = yes_logits / 5.0
            return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)

        if logits.ndim != 3:
            print(
                f"[RerankerService] WARNING: Causal LM logits shape {logits.shape} "
                f"is not 2D [batch,2] or 3D [batch,seq,vocab]. "
                f"Falling back to classification extraction."
            )
            return self._extract_classification_scores(logits, batch_size)

        attention_mask = ort_inputs.get("attention_mask")
        if attention_mask is not None:
            last_positions = attention_mask.sum(axis=1).astype(np.int64) - 1
        else:
            last_positions = np.full(
                logits.shape[0], logits.shape[1] - 1, dtype=np.int64
            )

        batch_indices = np.arange(logits.shape[0])
        last_token_logits = logits[batch_indices, last_positions]
        yes_logits = last_token_logits[:, self._yes_token_id].astype(np.float64)

        scaled = yes_logits / 5.0
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)

    @staticmethod
    def _extract_classification_scores(
        logits: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        if logits.ndim == 1:
            return logits.astype(np.float32)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            return logits[:, 0].astype(np.float32)
        elif logits.ndim == 2 and logits.shape[1] >= 2:
            logits_2d = logits.astype(np.float32)
            max_logits = np.max(logits_2d, axis=1, keepdims=True)
            exp_logits = np.exp(logits_2d - max_logits)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return probabilities[:, -1].astype(np.float32)
        else:
            logits_2d = logits.reshape(batch_size, -1).astype(np.float32)
            max_logits = np.max(logits_2d, axis=1, keepdims=True)
            exp_logits = np.exp(logits_2d - max_logits)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            relevant_class_idx = 1 if probabilities.shape[1] > 1 else 0
            return probabilities[:, relevant_class_idx].astype(np.float32)

    def _compute_scores(
        self,
        query: str,
        documents: List[Document],
    ) -> np.ndarray:
        if self.session is None or self.tokenizer is None:
            return np.asarray([], dtype=np.float32)

        with self._rerank_lock:
            normalized_query = _normalize_text_for_reranker(query)
            pair_count = len(documents)
            if pair_count == 0:
                return np.asarray([], dtype=np.float32)

            chunk_size = self.rerank_micro_batch_size
            if self.fixed_shape_padding and self.fixed_batch_size > 0:
                chunk_size = self.fixed_batch_size
            if chunk_size <= 0:
                chunk_size = pair_count

            input_names = {meta.name for meta in self.session.get_inputs()}
            all_scores: List[np.ndarray] = []
            _logged_first_batch = False

            for start in range(0, pair_count, chunk_size):
                end = min(pair_count, start + chunk_size)
                batch_docs = documents[start:end]
                pairs: List[Tuple[str, str]] = [
                    (
                        normalized_query,
                        _normalize_text_for_reranker(doc.page_content),
                    )
                    for doc in batch_docs
                ]

                real_count = len(pairs)
                if self.fixed_shape_padding and self.fixed_batch_size > 0:
                    target_batch = self.fixed_batch_size
                    if real_count < target_batch:
                        pad_count = target_batch - real_count
                        pairs.extend([(normalized_query, " ")] * pad_count)

                if self._is_causal_lm:
                    chat_texts = self._format_chat_pairs(pairs)
                    tokenized = self.tokenizer(
                        chat_texts,
                        padding="max_length" if self.fixed_shape_padding else True,
                        truncation=True,
                        max_length=self.rerank_max_length,
                        return_tensors="np",
                    )
                else:
                    tokenized = self.tokenizer(
                        pairs,
                        padding="max_length" if self.fixed_shape_padding else True,
                        truncation=True,
                        max_length=self.rerank_max_length,
                        return_tensors="np",
                    )

                ort_inputs = {}
                for key, value in tokenized.items():
                    if key not in input_names:
                        continue
                    if hasattr(value, "dtype") and value.dtype != np.int64:
                        value = value.astype(np.int64)
                    ort_inputs[key] = value

                if not _logged_first_batch and self.ort_log_severity_level == 0:
                    dtypes_summary = {k: v.dtype for k, v in ort_inputs.items()}
                    shapes_summary = {k: v.shape for k, v in ort_inputs.items()}
                    print(
                        f"[RerankerService] First batch diagnostics: "
                        f"dtypes={dtypes_summary}, shapes={shapes_summary}"
                    )

                outputs = self.session.run(None, ort_inputs)
                logits = np.asarray(outputs[0])

                if not _logged_first_batch and self.ort_log_severity_level == 0:
                    _logged_first_batch = True
                    print(
                        f"[RerankerService] First batch output: "
                        f"logits.shape={logits.shape}, logits.dtype={logits.dtype}, "
                        f"logits_sample={logits.ravel()[:4].tolist()}"
                    )

                if self._is_causal_lm and self._yes_token_id is not None:
                    batch_scores = self._extract_causal_lm_scores(
                        logits, ort_inputs, len(pairs)
                    )
                else:
                    batch_scores = self._extract_classification_scores(
                        logits, len(pairs)
                    )

                if self.fixed_shape_padding and self.fixed_batch_size > 0:
                    batch_scores = batch_scores[:real_count]

                all_scores.append(batch_scores)

            return np.concatenate(all_scores, axis=0)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        self.ensure_loaded()

        if not documents:
            return []

        if top_k <= 0:
            return []

        if self.session is None or self.tokenizer is None:
            return documents[:top_k]

        try:
            scores = self._compute_scores(query, documents)
            if scores.size == 0:
                return documents[:top_k]
        except Exception as exc:
            error_message = str(exc)
            if (
                self._is_tensorrt_runtime_failure(error_message)
                and not self._trt_demoted_to_cuda
                and not self._trt_runtime_disabled
                and not self.disable_runtime_cpu_fallback
            ):
                with self._bootstrap_lock:
                    if not self._trt_demoted_to_cuda:
                        print(
                            "[RerankerService] TensorRT runtime engine build failed; "
                            "demoting to CUDA+CPU provider chain for this process lifetime."
                        )
                        self._trt_demoted_to_cuda = True
                        self.session = None
                        try:
                            retry_sess_opt = ort.SessionOptions()
                            retry_sess_opt.intra_op_num_threads = 1
                            retry_sess_opt.inter_op_num_threads = 1
                            self._create_session_with_fallback(retry_sess_opt)
                        except Exception as retry_exc:
                            print(
                                "[RerankerService] Runtime fallback session init failed: "
                                f"{retry_exc}"
                            )
                            return documents[:top_k]

                try:
                    scores = self._compute_scores(query, documents)
                    if scores.size == 0:
                        return documents[:top_k]
                except Exception as retry_run_exc:
                    print(
                        f"[RerankerService] CPU fallback retry failed: {retry_run_exc}"
                    )
                    return documents[:top_k]
            else:
                print(f"[RerankerService] Rerank failed: {exc}")
                return documents[:top_k]

        scored_documents = list(zip(documents, scores.tolist()))
        scored_documents.sort(key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in scored_documents[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        self.ensure_loaded()

        if not documents:
            return []

        if top_k <= 0:
            return []

        if self.session is None or self.tokenizer is None:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            scores = self._compute_scores(query, documents)
            if scores.size == 0:
                return [(doc, 0.0) for doc in documents[:top_k]]
        except Exception as exc:
            error_message = str(exc)
            if (
                self._is_tensorrt_runtime_failure(error_message)
                and not self._trt_demoted_to_cuda
                and not self._trt_runtime_disabled
                and not self.disable_runtime_cpu_fallback
            ):
                with self._bootstrap_lock:
                    if not self._trt_demoted_to_cuda:
                        print(
                            "[RerankerService] TensorRT runtime engine build failed; "
                            "demoting to CUDA+CPU provider chain for this process lifetime."
                        )
                        self._trt_demoted_to_cuda = True
                        self.session = None
                        try:
                            retry_sess_opt = ort.SessionOptions()
                            retry_sess_opt.intra_op_num_threads = 1
                            retry_sess_opt.inter_op_num_threads = 1
                            self._create_session_with_fallback(retry_sess_opt)
                        except Exception as retry_exc:
                            print(
                                "[RerankerService] Runtime fallback session init failed: "
                                f"{retry_exc}"
                            )
                            return [(doc, 0.0) for doc in documents[:top_k]]

                try:
                    scores = self._compute_scores(query, documents)
                    if scores.size == 0:
                        return [(doc, 0.0) for doc in documents[:top_k]]
                except Exception as retry_run_exc:
                    print(
                        f"[RerankerService] CPU fallback retry failed: {retry_run_exc}"
                    )
                    return [(doc, 0.0) for doc in documents[:top_k]]
            else:
                print(f"[RerankerService] Rerank failed: {exc}")
                return [(doc, 0.0) for doc in documents[:top_k]]

        scored_documents = list(zip(documents, scores.tolist()))
        scored_documents.sort(key=lambda item: float(item[1]), reverse=True)
        return [(doc, float(score)) for doc, score in scored_documents[:top_k]]


def _normalize_text_for_reranker(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    normalized = "".join(
        ch for ch in normalized if ch in {"\n", "\r", "\t"} or ch.isprintable()
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# Global singleton accessor
def get_reranker_service() -> RerankerService:
    return RerankerService()


if __name__ == "__main__":
    service = get_reranker_service()
    service.ensure_loaded()

    docs = [
        Document(
            page_content="Contract law governs legally binding agreements.",
            metadata={"id": 1},
        ),
        Document(
            page_content="Criminal law defines offenses and penalties.",
            metadata={"id": 2},
        ),
        Document(
            page_content="Tort law addresses civil wrongs and liabilities.",
            metadata={"id": 3},
        ),
    ]

    query_text = "What law applies to breaches of agreement?"
    reranked = service.rerank(query_text, docs, top_k=2)

    print("[RerankerService] Rerank test results:")
    for idx, doc in enumerate(reranked, 1):
        print(f"{idx}. id={doc.metadata.get('id')} content={doc.page_content}")
