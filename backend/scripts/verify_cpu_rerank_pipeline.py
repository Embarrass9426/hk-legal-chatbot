import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dotenv import load_dotenv

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env

setup_env.setup_cuda_dlls()

from backend.core.utils import generate_hyde_embeddings
from backend.services.embedding_service import get_embedding_service
from backend.services.qdrant_store import QdrantStoreManager
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def _count_missing_metadata_fields(docs: List[Any]) -> int:
    missing = 0
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if not metadata.get("doc_id") or not metadata.get("section_id"):
            missing += 1
    return missing


def _count_empty_content_docs(docs: List[Any]) -> int:
    empty = 0
    for doc in docs:
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            empty += 1
    return empty


def _count_expanded_chunks(expanded: List[Dict[str, Any]]) -> int:
    total = 0
    for section in expanded:
        total += len(section.get("chunks", []))
    return total


def _count_regrouped_chunks(grouped: List[Dict[str, Any]]) -> int:
    total = 0
    for section in grouped:
        total += len(section.get("chunks", []))
    return total


def _build_judge_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    return ChatOpenAI(
        model=os.getenv("HK_LEGAL_EVAL_JUDGE_MODEL", "deepseek-chat"),
        api_key=SecretStr(api_key),
        base_url="https://api.deepseek.com",
        temperature=0,
    )


async def _generate_hyde_embedding(query: str) -> List[float]:
    judge_llm = _build_judge_llm()
    embedding_service = get_embedding_service()
    embedding_service.ensure_loaded()
    return await generate_hyde_embeddings(
        user_query=query,
        llm=judge_llm,
        embedding_service=embedding_service,
    )


def _generate_query_embedding_fast(query: str) -> List[float]:
    embedding_service = get_embedding_service()
    embedding_service.ensure_loaded()
    query_with_prefix = (
        f"Represent this question for retrieving relevant legal documents: {query}"
    )
    return embedding_service.embed_query(query_with_prefix)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that hyde+rerrank pipeline retrieves chunks when reranker runs on CPU"
    )
    parser.add_argument("--query", required=True, help="User query to test")
    parser.add_argument("--vector-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--max-chunks-per-section", type=int, default=20)
    parser.add_argument(
        "--verify-mode",
        choices=["full", "pipeline_only"],
        default="pipeline_only",
        help="pipeline_only skips heavy rerank scoring and verifies retrieval/expansion/regroup viability",
    )
    parser.add_argument(
        "--rerank-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for explicit rerank verification stage",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "hyde"],
        default="fast",
        help="fast=direct query embedding (no LLM), hyde=full HyDE generation",
    )
    args = parser.parse_args()

    backend_env = os.path.join(project_root, "backend", ".env")
    load_dotenv(backend_env)

    os.environ["RERANKER_FORCE_CPU"] = "1"
    os.environ["RERANKER_REQUIRE_TENSORRT"] = "0"

    vs_manager = QdrantStoreManager()
    reranker = vs_manager.reranker
    if reranker is None:
        print("FAIL: reranker disabled (ENABLE_RERANKER=0)")
        return 10

    reranker.configure_model(
        model_dir_name=os.getenv("RERANKER_MODEL_DIR", "zerank-1-small"),
        onnx_model_file=os.getenv("RERANKER_ONNX_FILE", "model_fp16.onnx"),
    )

    smoke_docs = [
        Document(
            page_content="alpha legal chunk",
            metadata={"doc_id": "s", "section_id": "1"},
        ),
        Document(
            page_content="beta legal chunk", metadata={"doc_id": "s", "section_id": "2"}
        ),
    ]
    smoke_out = reranker.rerank("test query", smoke_docs, top_k=1)
    print(f"cpu_rerank_smoke_count={len(smoke_out)}")
    if len(smoke_out) == 0:
        print("FAIL: CPU reranker smoke test returned 0")
        return 10

    if args.mode == "hyde":
        print("embedding_mode=hyde")
        hyde_embedding = asyncio.run(_generate_hyde_embedding(args.query))
    else:
        print("embedding_mode=fast")
        hyde_embedding = _generate_query_embedding_fast(args.query)

    matches: Any = vs_manager.index.query(
        vector=hyde_embedding,
        top_k=args.vector_top_k,
        include_metadata=True,
        include_values=False,
    )

    initial_docs: List[Document] = []
    for match in matches.get("matches", []):
        metadata = match.get("metadata") or {}
        content = metadata.get("content", "")
        initial_docs.append(Document(page_content=content, metadata=metadata))

    pinecone_matches = len(initial_docs)
    pinecone_empty_content = _count_empty_content_docs(initial_docs)
    pinecone_missing_meta = _count_missing_metadata_fields(initial_docs)
    print(f"pinecone_matches={pinecone_matches}")
    print(f"pinecone_empty_content={pinecone_empty_content}")
    print(f"pinecone_missing_meta={pinecone_missing_meta}")

    if pinecone_matches == 0:
        print("FAIL: pinecone returned 0 matches")
        return 11

    expanded = vs_manager._expand_to_sections(
        initial_docs,
        max_chunks_per_section=args.max_chunks_per_section,
    )
    expanded_groups = len(expanded)
    expanded_chunks = _count_expanded_chunks(expanded)
    print(f"expanded_groups={expanded_groups}")
    print(f"expanded_chunks={expanded_chunks}")
    if expanded_chunks == 0:
        print("FAIL: expansion produced 0 chunks")
        return 13

    flat_docs = vs_manager._flatten_expanded_sections(expanded)
    flat_count = len(flat_docs)
    flat_empty = _count_empty_content_docs(flat_docs)
    flat_missing_meta = _count_missing_metadata_fields(flat_docs)
    print(f"flattened_docs={flat_count}")
    print(f"flattened_empty_content={flat_empty}")
    print(f"flattened_missing_meta={flat_missing_meta}")
    if flat_count == 0:
        print("FAIL: flatten produced 0 docs")
        return 14

    if args.verify_mode == "pipeline_only":
        pipeline_only_sections = vs_manager._regroup_to_sections(
            flat_docs[: args.rerank_top_k]
        )
        pipeline_only_chunks = _count_regrouped_chunks(pipeline_only_sections)
        print(f"pipeline_only_sections={len(pipeline_only_sections)}")
        print(f"pipeline_only_chunks={pipeline_only_chunks}")
        if pipeline_only_chunks == 0:
            print("FAIL: pipeline-only regroup returned 0 chunks")
            return 16

        print(
            "PASS: Retrieval/expansion/regroup pipeline returns chunks (CPU rerank skipped)"
        )
        return 0

    start_ts = time.time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            reranker.rerank, args.query, flat_docs, args.rerank_top_k
        )
        try:
            reranked = future.result(timeout=args.rerank_timeout_seconds)
        except FuturesTimeoutError:
            print(
                "FAIL: reranker stage timed out after "
                f"{args.rerank_timeout_seconds}s on {flat_count} flattened docs"
            )
            return 18
    print(f"rerank_elapsed_seconds={time.time() - start_ts:.2f}")

    reranked_count = len(reranked)
    print(f"reranked_docs={reranked_count}")
    if reranked_count == 0:
        print("FAIL: reranker returned 0 docs")
        return 15

    grouped = vs_manager._regroup_to_sections(reranked)
    grouped_sections = len(grouped)
    grouped_chunks = _count_regrouped_chunks(grouped)
    print(f"regrouped_sections={grouped_sections}")
    print(f"regrouped_chunks={grouped_chunks}")
    if grouped_chunks == 0:
        print("FAIL: regroup produced 0 chunks")
        return 16

    final_pipeline = vs_manager.search_hyde_with_rerank_and_expansion(
        hyde_embedding=hyde_embedding,
        original_query=args.query,
        k=args.vector_top_k,
        rerank_top_k=args.rerank_top_k,
        max_chunks_per_section=args.max_chunks_per_section,
    )
    final_sections = len(final_pipeline)
    final_chunks = _count_regrouped_chunks(final_pipeline)
    print(f"final_sections={final_sections}")
    print(f"final_chunks={final_chunks}")
    if final_chunks == 0:
        print("FAIL: final public pipeline returned 0 chunks")
        return 17

    print("PASS: CPU reranker pipeline retrieved chunks successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
