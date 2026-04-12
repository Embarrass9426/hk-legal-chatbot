import os
import sys
from typing import List

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env

setup_env.setup_cuda_dlls()

from langchain_core.documents import Document
from backend.services.reranker_service import get_reranker_service


def _assert_distinct(results: List[Document], expected_first_id: str) -> None:
    if not results:
        raise AssertionError("Reranker returned no results")

    first_id = str(results[0].metadata.get("id", ""))
    if first_id != expected_first_id:
        raise AssertionError(
            f"Unexpected top document id: got={first_id!r}, expected={expected_first_id!r}"
        )


def run_test() -> None:
    os.environ.setdefault("RERANKER_STRICT_ONNX_FILE", "0")
    service = get_reranker_service()
    service.ensure_loaded()

    query = "Can I claim compensation for a workplace injury after a falling metal bar?"
    noisy_query = query + " \u200b\u00a0\ufeff\u2060"

    docs = [
        Document(
            page_content=(
                "Employees injured in workplace accidents may claim compensation for "
                "medical expenses and wage loss under relevant ordinance provisions."
            ),
            metadata={"id": "relevant"},
        ),
        Document(
            page_content=(
                "This passage discusses unrelated licensing procedures for street vendors "
                "and does not address workplace injury compensation."
            ),
            metadata={"id": "irrelevant"},
        ),
    ]

    normal_results = service.rerank(query, docs, top_k=2)
    noisy_results = service.rerank(noisy_query, docs, top_k=2)

    _assert_distinct(normal_results, "relevant")
    _assert_distinct(noisy_results, "relevant")

    if service.is_loaded():
        service.unload()

    print("[test_reranker_tokenizer_unicode] PASS")


if __name__ == "__main__":
    run_test()
