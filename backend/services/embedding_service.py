import os
import sys
import threading
import time

# MUST setup CUDA DLLs before importing torch/onnxruntime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_env

setup_env.setup_cuda_dlls()

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

        self.model_path = self._get_model_path()
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

        # 2. Configure Providers (Priority: TensorRT -> CUDA -> CPU)
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1

        providers_config = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        print("[EmbeddingService] Initializing ORT with TensorRT provider")

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_path,
            providers=providers_config,
            session_options=sess_opt,
            trust_remote_code=True,
        )

        # Log active providers
        if hasattr(self.model, "model"):
            print(
                f"[EmbeddingService] Active Providers: {self.model.model.get_providers()}"
            )
            self.model.use_io_binding = True

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
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
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
                    embeddings[zero_rows] += (
                        torch.rand_like(embeddings[zero_rows]) * 1e-6
                    )

                # Convert to list
                return embeddings.cpu().numpy().tolist()

            except Exception as e:
                error_msg = str(e)
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
