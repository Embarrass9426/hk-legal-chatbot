import os
import sys
import threading
import time
import warnings

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env

setup_env.setup_cuda_dlls()

warnings.filterwarnings(
    "ignore",
    message=r".*CUDA capability.*not compatible with the current PyTorch installation.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*",
    category=UserWarning,
    module=r"torch\.cuda(\..*)?",
)

import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from typing import List, Optional


class EmbeddingService:
    _instance = None
    _init_lock = threading.Lock()
    _bootstrap_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    print(
                        f"[EmbeddingService] Creating NEW singleton instance (ID: {id(cls)})"
                    )
                    cls._instance = super(EmbeddingService, cls).__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._embed_lock = threading.Lock()
        else:
            print(
                f"[EmbeddingService] Reusing existing singleton instance (ID: {id(cls._instance)})"
            )
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._bootstrap_lock:
            if self._initialized:
                return

            self.model_path = self._get_model_path()
            self.tokenizer = None
            self.model = None
            self.device = "cpu"
            self.expected_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
            self.require_tensorrt = os.getenv(
                "EMBEDDING_REQUIRE_TENSORRT", "1"
            ).strip().lower() not in {"0", "false", "no"}
            self.trt_fp16_preferred = os.getenv(
                "EMBEDDING_TRT_FP16", "1"
            ).strip().lower() not in {"0", "false", "no"}
            self.strict_fp16 = os.getenv(
                "EMBEDDING_STRICT_FP16", "1"
            ).strip().lower() not in {
                "0",
                "false",
                "no",
            }
            self._active_fp16 = self.trt_fp16_preferred
            self.trt_max_workspace_size = int(
                os.getenv("EMBEDDING_TRT_MAX_WORKSPACE_SIZE", "268435456")
            )
            self.trt_min_subgraph_size = int(
                os.getenv("EMBEDDING_TRT_MIN_SUBGRAPH_SIZE", "5")
            )
            self.trt_max_partition_iterations = int(
                os.getenv("EMBEDDING_TRT_MAX_PARTITION_ITERATIONS", "200")
            )
            self.trt_auxiliary_streams = int(
                os.getenv("EMBEDDING_TRT_AUX_STREAMS", "0")
            )
            self._load_model()
            self._initialized = True

    def _get_model_path(self) -> str:
        # Resolve path relative to this file: backend/services/embedding_service.py
        # Model is at: backend/models/yuan-onnx-trt
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "models", "yuan-onnx-trt")

    @classmethod
    def clear_tensorrt_cache(cls):
        """Clear TensorRT engine cache. Call before first load if issues occur."""
        import shutil

        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "yuan-onnx-trt",
            "cache",
        )
        if os.path.exists(cache_dir):
            print(f"[EmbeddingService] Clearing TensorRT cache: {cache_dir}")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print("[EmbeddingService] Cache cleared.")

    def _load_model(self):
        print(f"[EmbeddingService] Loading model from: {self.model_path}")

        # 1. Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, fix_mistral_regex=True
            )
        except Exception as e:
            print(f"[EmbeddingService] Failed to load tokenizer: {e}")
            raise

        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1

        trt_fp16_enable = self._active_fp16
        trt_cache_path = os.path.join(self.model_path, "cache")
        os.makedirs(trt_cache_path, exist_ok=True)

        available_providers = ort.get_available_providers()
        has_tensorrt_provider = "TensorrtExecutionProvider" in available_providers

        if self.require_tensorrt and not has_tensorrt_provider:
            raise RuntimeError(
                "TensorRT provider is required but unavailable in this runtime. "
                "Set EMBEDDING_REQUIRE_TENSORRT=0 to allow CPU fallback or install "
                "a runtime exposing TensorrtExecutionProvider. "
                f"Available providers: {available_providers}"
            )

        use_tensorrt = has_tensorrt_provider

        if use_tensorrt:
            providers_config = [
                "TensorrtExecutionProvider",
                "CPUExecutionProvider",
            ]
            provider_options = [
                {
                    "trt_fp16_enable": trt_fp16_enable,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache_path,
                    "trt_layer_norm_fp32_fallback": True,
                    "trt_max_workspace_size": self.trt_max_workspace_size,
                    "trt_min_subgraph_size": self.trt_min_subgraph_size,
                    "trt_max_partition_iterations": self.trt_max_partition_iterations,
                    "trt_auxiliary_streams": self.trt_auxiliary_streams,
                    "trt_context_memory_sharing_enable": True,
                    "trt_builder_optimization_level": 2,
                    "trt_force_sequential_engine_build": True,
                },
                {},
            ]
            print("[EmbeddingService] Initializing ORT with TensorRT provider")
            print(f"[EmbeddingService] TensorRT FP16 enabled: {trt_fp16_enable}")
            print(
                "[EmbeddingService] TensorRT opts: "
                f"workspace={self.trt_max_workspace_size}, "
                f"min_subgraph={self.trt_min_subgraph_size}, "
                f"max_partition_iterations={self.trt_max_partition_iterations}, "
                f"aux_streams={self.trt_auxiliary_streams}"
            )
        else:
            providers_config = ["CPUExecutionProvider"]
            provider_options = [{}]
            print(
                "[EmbeddingService] TensorrtExecutionProvider unavailable. "
                "Falling back to CPUExecutionProvider."
            )

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_path,
            providers=providers_config,
            provider_options=provider_options,
            session_options=sess_opt,
            trust_remote_code=True,
        )

        # Log active providers
        if hasattr(self.model, "model"):
            active_providers = self.model.model.get_providers()
            print(f"[EmbeddingService] Active Providers: {active_providers}")

            if (
                self.require_tensorrt
                and "TensorrtExecutionProvider" not in active_providers
            ):
                raise RuntimeError(
                    "TensorRT provider is required but not active. "
                    "Install/verify onnxruntime-gpu + TensorRT libraries in WSL and ensure "
                    "backend/core/setup_env.py can preload tensorrt_libs."
                )

            if "CUDAExecutionProvider" in active_providers:
                raise RuntimeError(
                    "CUDAExecutionProvider is active, but this project is configured to avoid CUDA EP. "
                    "Please remove CUDA EP from runtime provider chain."
                )

            self.model.use_io_binding = False

    def _switch_to_fp32_tensorrt(self):
        if not self._active_fp16:
            return

        print(
            "[EmbeddingService] FP16 instability detected. Switching TensorRT to FP32."
        )
        self._active_fp16 = False
        self._load_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Thread-safe.
        """
        # Handle empty/whitespace texts to prevent crashes
        clean_texts = [t if t.strip() else "." for t in texts]

        with self._embed_lock:
            return self._embed_batch(clean_texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        """
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with retry on transient TensorRT errors."""
        MAX_RETRIES = 3
        RETRY_DELAY = [0.5, 1.0, 2.0]  # Exponential backoff

        for attempt in range(MAX_RETRIES):
            try:
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Add position_ids if missing
                if "position_ids" not in inputs:
                    inputs["position_ids"] = (
                        torch.arange(0, inputs["input_ids"].size(1), dtype=torch.long)
                        .unsqueeze(0)
                        .expand_as(inputs["input_ids"])
                    )

                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Mean Pooling
                last_hidden = outputs.last_hidden_state.detach().cpu()
                attention_mask = inputs["attention_mask"].detach().cpu()
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = torch.nan_to_num(
                    embeddings, nan=0.0, posinf=0.0, neginf=0.0
                )

                # Validate (catch all-zero bug)
                zero_rows = embeddings.abs().sum(dim=1) == 0
                if zero_rows.all():
                    raise RuntimeError(
                        f"Embedding model produced all-zero vectors for entire batch "
                        f"({embeddings.shape[0]} texts). Model may be broken — aborting."
                    )
                if zero_rows.any():
                    if self.strict_fp16:
                        raise RuntimeError(
                            f"Embedding produced {int(zero_rows.sum().item())} zero vectors in strict FP16 mode."
                        )

                    embeddings[zero_rows] += 1e-6

                # Convert to list
                vectors = embeddings.cpu().numpy().astype(np.float32).tolist()

                for idx, vec in enumerate(vectors):
                    if len(vec) != self.expected_dimension:
                        raise RuntimeError(
                            f"Embedding dimension mismatch at row {idx}: "
                            f"got {len(vec)}, expected {self.expected_dimension}."
                        )

                    l2 = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
                    if not np.isfinite(l2) or l2 <= 1e-8:
                        raise RuntimeError(
                            f"Invalid embedding norm at row {idx}: {l2}."
                        )

                return vectors

            except Exception as e:
                error_msg = str(e)

                validation_failure = (
                    "all-zero vectors" in error_msg
                    or "strict fp16 mode" in error_msg.lower()
                    or "Invalid embedding norm" in error_msg
                    or "dimension mismatch" in error_msg.lower()
                )

                if self._active_fp16 and validation_failure and not self.strict_fp16:
                    self._switch_to_fp32_tensorrt()
                    continue

                is_transient = (
                    "enqueueV3" in error_msg
                    or "TensorRT EP" in error_msg
                    or "Cuda Runtime" in error_msg
                )

                if is_transient and attempt < MAX_RETRIES - 1:
                    print(
                        f"[EmbeddingService] TensorRT error (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg[:100]}..."
                    )
                    print(f"[EmbeddingService] Retrying in {RETRY_DELAY[attempt]}s...")
                    time.sleep(RETRY_DELAY[attempt])
                    continue
                else:
                    # Persistent failure or non-transient error
                    if is_transient:
                        print(
                            f"[EmbeddingService] PERSISTENT ERROR after {attempt + 1} attempts"
                        )
                        print(f"[EmbeddingService] Clearing TensorRT cache...")
                        EmbeddingService.clear_tensorrt_cache()
                        raise RuntimeError(
                            f"Embedding failed after {MAX_RETRIES} retries: {error_msg}"
                        )
                    else:
                        # Non-transient error (e.g., all-zero embeddings) - fail immediately
                        raise


# Global singleton accessor
def get_embedding_service():
    return EmbeddingService()
