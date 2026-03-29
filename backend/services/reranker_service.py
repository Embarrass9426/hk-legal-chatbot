import os
import sys
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
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


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
        if self._initialized:
            return

        with self._bootstrap_lock:
            if self._initialized:
                return

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = None

            try:
                print("[RerankerService] Loading model...")
                self.model = CrossEncoder(
                    "zeroentropy/zerank-2",
                    trust_remote_code=True,
                    device=self.device,
                )
                print("[RerankerService] Model loaded successfully")
            except Exception as exc:
                print(f"[RerankerService] Failed to load model: {exc}")
                self.model = None

            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        if not documents:
            return []

        if top_k <= 0:
            return []

        if self.model is None:
            return documents[:top_k]

        pairs = [(query, doc.page_content) for doc in documents]

        try:
            with self._rerank_lock:
                scores = self.model.predict(pairs)
        except Exception as exc:
            print(f"[RerankerService] Rerank failed: {exc}")
            return documents[:top_k]

        scored_documents = list(zip(documents, scores))
        scored_documents.sort(key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in scored_documents[:top_k]]


# Global singleton accessor
def get_reranker_service() -> RerankerService:
    return RerankerService()


if __name__ == "__main__":
    service = get_reranker_service()

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
