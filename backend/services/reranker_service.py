import os
import sys
import gc
import threading
from typing import List

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
            self.model_path = os.path.join(base_dir, "models", "zerank-1-small-q8")
            self.onnx_model_file = "model_q8.onnx"
            self.tokenizer = None
            self.session = None
            self._model_loaded = False
            self.trt_engine_cache_enable = True
            self.trt_engine_cache_path = os.path.join(self.model_path, "cache")
            self.trt_max_workspace_size = 268435456

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
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model_loaded = False
            print("[RerankerService] Model unloaded.")

    def _load_model(self):
        print(f"[RerankerService] Loading model from: {self.model_path}")

        onnx_model_path = os.path.join(self.model_path, self.onnx_model_file)
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[RerankerService] Failed to load tokenizer: {e}")
            raise

        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1

        os.makedirs(self.trt_engine_cache_path, exist_ok=True)

        available_providers = ort.get_available_providers()
        has_tensorrt_provider = "TensorrtExecutionProvider" in available_providers

        if has_tensorrt_provider:
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": self.trt_engine_cache_enable,
                        "trt_engine_cache_path": self.trt_engine_cache_path,
                        "trt_max_workspace_size": self.trt_max_workspace_size,
                    },
                ),
                "CPUExecutionProvider",
            ]
            print("[RerankerService] Initializing ORT with TensorRT provider")
        else:
            providers = ["CPUExecutionProvider"]
            print(
                "[RerankerService] TensorrtExecutionProvider unavailable. "
                "Falling back to CPUExecutionProvider."
            )

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_opt,
            providers=providers,
        )

        active_providers = self.session.get_providers()
        print(f"[RerankerService] Active Providers: {active_providers}")

        if "CUDAExecutionProvider" in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is active, but this project is configured to avoid CUDA EP. "
                "Please remove CUDA EP from runtime provider chain."
            )

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
            with self._rerank_lock:
                pairs = [(query, doc.page_content) for doc in documents]
                tokenized = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )

                input_names = {meta.name for meta in self.session.get_inputs()}
                ort_inputs = {
                    key: value for key, value in tokenized.items() if key in input_names
                }

                outputs = self.session.run(None, ort_inputs)
                logits = np.asarray(outputs[0])

                if logits.ndim == 1:
                    scores = logits.astype(np.float32)
                elif logits.ndim == 2 and logits.shape[1] == 1:
                    scores = logits[:, 0].astype(np.float32)
                else:
                    logits_2d = logits.reshape(len(pairs), -1).astype(np.float32)
                    max_logits = np.max(logits_2d, axis=1, keepdims=True)
                    exp_logits = np.exp(logits_2d - max_logits)
                    probabilities = exp_logits / np.sum(
                        exp_logits, axis=1, keepdims=True
                    )
                    relevant_class_idx = 1 if probabilities.shape[1] > 1 else 0
                    scores = probabilities[:, relevant_class_idx].astype(np.float32)
        except Exception as exc:
            print(f"[RerankerService] Rerank failed: {exc}")
            return documents[:top_k]

        scored_documents = list(zip(documents, scores.tolist()))
        scored_documents.sort(key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in scored_documents[:top_k]]


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
