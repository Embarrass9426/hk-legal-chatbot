import os
import sys
import json
import asyncio
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env

setup_env.setup_cuda_dlls()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.vector_store import VectorStoreManager
from backend.core.utils import rewrite_query

load_dotenv()


def parse_judge_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def safe_judge_parse(text: str) -> Dict[str, Any]:
    try:
        parsed = parse_judge_response(text)
        score = float(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
        score = max(1.0, min(10.0, score))
        score = round(score, 1)
        return {"score": score, "reasoning": reasoning}
    except Exception as exc:
        return {"score": 0, "reasoning": f"Parse error: {exc}"}


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _build_doc_section_key(metadata: Dict[str, Any]) -> Tuple[str, str]:
    return (
        _safe_str(metadata.get("doc_id", "Unknown"), "Unknown"),
        _safe_str(metadata.get("section_id", "Unknown"), "Unknown"),
    )


def collapse_documents_to_sections(results: List[Any]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for doc in results:
        metadata = getattr(doc, "metadata", {}) or {}
        content = _safe_str(getattr(doc, "page_content", ""), "")
        content = content.strip()
        if not content:
            continue

        key = _build_doc_section_key(metadata)
        if key not in grouped:
            grouped[key] = {
                "doc_id": key[0],
                "section_id": key[1],
                "section_title": _safe_str(
                    metadata.get("section_title", "Unknown Section"),
                    "Unknown Section",
                ),
                "citation": _safe_str(
                    metadata.get("citation", f"Section {key[1]}"),
                    f"Section {key[1]}",
                ),
                "source_url": _safe_str(metadata.get("source_url", ""), ""),
                "pages": set(),
                "content_parts": [],
            }

        page_number = metadata.get("page_number")
        if page_number not in (None, ""):
            grouped[key]["pages"].add(_safe_str(page_number))

        grouped[key]["content_parts"].append(content)

    sections: List[Dict[str, Any]] = []
    for _, section in grouped.items():
        unique_parts: List[str] = []
        seen = set()
        for part in section["content_parts"]:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        pages = sorted(section["pages"], key=lambda x: (len(x), x))
        full_content = "\n\n".join(unique_parts)

        sections.append(
            {
                "doc_id": section["doc_id"],
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "citation": section["citation"],
                "source_url": section["source_url"],
                "pages": pages,
                "content": full_content,
            }
        )

    sections.sort(key=lambda s: (s["doc_id"], s["section_id"]))
    return sections


def collapse_expanded_sections(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []

    for section in results:
        chunks = section.get("chunks", [])
        if not chunks:
            continue

        first_metadata = chunks[0].get("metadata", {}) or {}
        doc_id = _safe_str(
            section.get("doc_id", first_metadata.get("doc_id", "Unknown"))
        )
        section_id = _safe_str(
            section.get("section_id", first_metadata.get("section_id", "Unknown"))
        )
        section_title = _safe_str(
            first_metadata.get("section_title", "Unknown Section"),
            "Unknown Section",
        )
        citation = _safe_str(
            first_metadata.get("citation", f"Section {section_id}"),
            f"Section {section_id}",
        )
        source_url = _safe_str(first_metadata.get("source_url", ""), "")

        sorted_chunks = sorted(
            chunks,
            key=lambda c: int((c.get("metadata", {}) or {}).get("chunk_index", 0)),
        )

        pages_set = set()
        content_parts: List[str] = []
        for chunk in sorted_chunks:
            metadata = chunk.get("metadata", {}) or {}
            page_number = metadata.get("page_number")
            if page_number not in (None, ""):
                pages_set.add(_safe_str(page_number))

            chunk_content = _safe_str(chunk.get("content", ""), "").strip()
            if chunk_content:
                content_parts.append(chunk_content)

        if not content_parts:
            continue

        sections.append(
            {
                "doc_id": doc_id,
                "section_id": section_id,
                "section_title": section_title,
                "citation": citation,
                "source_url": source_url,
                "pages": sorted(pages_set, key=lambda x: (len(x), x)),
                "content": "\n\n".join(content_parts),
            }
        )

    sections.sort(key=lambda s: (s["doc_id"], s["section_id"]))
    return sections


def build_context_text_from_sections(sections: List[Dict[str, Any]]) -> str:
    context_parts: List[str] = []
    for i, section in enumerate(sections):
        pages = section.get("pages", [])
        pages_text = ", ".join(pages) if pages else "Unknown"
        context_parts.append(
            f"--- Source {i + 1} ---\n"
            f"Cap: {section.get('doc_id', 'Unknown')}\n"
            f"Citation: {section.get('citation', 'Unknown')}\n"
            f"Section Title: {section.get('section_title', 'Unknown')}\n"
            f"Pages: {pages_text}\n"
            f"Content: {section.get('content', '')}"
        )
    return "\n\n".join(context_parts)


async def assess_sources_usefulness(
    query: str,
    answer: str,
    sections: List[Dict[str, Any]],
    llm: ChatOpenAI,
) -> List[Dict[str, Any]]:
    if not sections:
        return []

    sources_payload: List[Dict[str, Any]] = []
    for idx, section in enumerate(sections, start=1):
        sources_payload.append(
            {
                "source_index": idx,
                "doc_id": section.get("doc_id", "Unknown"),
                "citation": section.get("citation", "Unknown"),
                "section_title": section.get("section_title", "Unknown"),
                "content": section.get("content", ""),
            }
        )

    system_prompt = (
        "You are evaluating retrieved legal sources for usefulness. "
        "For EACH source, decide if it is useful for answering the user's question "
        "and whether the final answer should rely on it. "
        "Use a usefulness_score from 0.0 to 10.0 with exactly one decimal."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Final Answer: {answer}\n\n"
        f"Sources (JSON):\n{json.dumps(sources_payload, ensure_ascii=False)}\n\n"
        "Return JSON ONLY in this exact schema:\n"
        "{\n"
        '  "sources": [\n'
        "    {\n"
        '      "source_index": 1,\n'
        '      "is_useful": true,\n'
        '      "usefulness_score": 8.5,\n'
        '      "reasoning": "short explanation"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules: one object per source_index, no omissions, one decimal place for usefulness_score."
    )

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = parse_judge_response(_safe_str(response.content, ""))
        source_items = parsed.get("sources", [])

        normalized: List[Dict[str, Any]] = []
        mapped = {
            int(item.get("source_index", -1)): item
            for item in source_items
            if isinstance(item, dict)
        }

        for idx, section in enumerate(sections, start=1):
            item = mapped.get(idx, {})
            raw_score = item.get("usefulness_score", 0)
            try:
                numeric_score = round(max(0.0, min(10.0, float(raw_score))), 1)
            except Exception:
                numeric_score = 0.0

            normalized.append(
                {
                    "source_index": idx,
                    "doc_id": section.get("doc_id", "Unknown"),
                    "citation": section.get("citation", "Unknown"),
                    "section_title": section.get("section_title", "Unknown"),
                    "is_useful": bool(item.get("is_useful", False)),
                    "usefulness_score": numeric_score,
                    "reasoning": _safe_str(item.get("reasoning", ""), ""),
                }
            )

        return normalized
    except Exception as exc:
        fallback: List[Dict[str, Any]] = []
        for idx, section in enumerate(sections, start=1):
            fallback.append(
                {
                    "source_index": idx,
                    "doc_id": section.get("doc_id", "Unknown"),
                    "citation": section.get("citation", "Unknown"),
                    "section_title": section.get("section_title", "Unknown"),
                    "is_useful": False,
                    "usefulness_score": 0.0,
                    "reasoning": f"Source usefulness parse error: {exc}",
                }
            )
        return fallback


async def generate_answer(query: str, context_text: str, llm: ChatOpenAI) -> str:
    system_prompt = (
        "You are an expert Hong Kong legal assistant. "
        "Answer the user's question based ONLY on the provided legal context. "
        "If the context doesn't contain enough information, say so. "
        "Always cite specific sections when possible."
    )

    messages = [
        SystemMessage(content=system_prompt + "\n\nCONTEXT:\n" + context_text),
        HumanMessage(content=query),
    ]
    response = await llm.ainvoke(messages)
    return str(response.content).strip()


async def judge_relevance(query: str, answer: str, llm: ChatOpenAI) -> Dict[str, Any]:
    system_prompt = (
        "You are evaluating whether an AI answer addresses the user's legal "
        "question. Score 1.0-10.0 with one decimal place. "
        "10.0=perfectly addresses the question. "
        "1=completely irrelevant."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        'Respond in JSON: {"score": <1.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}


async def judge_groundedness(
    answer: str, context_text: str, llm: ChatOpenAI
) -> Dict[str, Any]:
    system_prompt = (
        "You are evaluating whether an AI answer is grounded in the provided "
        "legal context. Score 1.0-10.0 with one decimal place. "
        "10.0=every claim is supported by the context. "
        "1=answer contradicts or fabricates information not in context."
    )
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Answer: {answer}\n\n"
        'Respond in JSON: {"score": <1.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}


async def judge_retrieval_relevance(
    query: str, context_text: str, llm: ChatOpenAI
) -> Dict[str, Any]:
    system_prompt = (
        "You are evaluating whether retrieved legal documents are relevant to the "
        "user's question. Score 1.0-10.0 with one decimal place. "
        "10.0=all retrieved content is highly relevant. "
        "1=none of the content relates to the question."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        'Respond in JSON: {"score": <1.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}


async def evaluate_strategy(
    strategy_name: str,
    query: str,
    context_text: str,
    answer: str,
    llm: ChatOpenAI,
) -> Dict[str, Any]:
    relevance = await judge_relevance(query, answer, llm)
    groundedness = await judge_groundedness(answer, context_text, llm)
    retrieval_relevance = await judge_retrieval_relevance(query, context_text, llm)
    avg_score = (
        relevance["score"] + groundedness["score"] + retrieval_relevance["score"]
    ) / 3
    return {
        "relevance": relevance,
        "groundedness": groundedness,
        "retrieval_relevance": retrieval_relevance,
        "avg_score": round(avg_score, 1),
    }


def make_failed_strategy_result(error_message: str) -> Dict[str, Any]:
    failed_score = {"score": 0, "reasoning": error_message}
    return {
        "answer": "",
        "num_sources": 0,
        "retrieved_sections": [],
        "source_usefulness": [],
        "scores": {
            "relevance": failed_score,
            "groundedness": failed_score,
            "retrieval_relevance": failed_score,
            "avg_score": 0,
        },
    }


def compute_strategy_summary(
    details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    strategy_names = [
        "plain_vector",
        "rewritten_vector",
        "rewritten_expanded",
        "rewritten_rerank_expanded",
    ]
    summary: Dict[str, Dict[str, float]] = {}

    for strategy in strategy_names:
        relevance_scores: List[float] = []
        groundedness_scores: List[float] = []
        retrieval_relevance_scores: List[float] = []
        overall_scores: List[float] = []

        for item in details:
            strategy_data = item.get("strategies", {}).get(strategy, {})
            scores = strategy_data.get("scores", {})
            relevance_scores.append(float(scores.get("relevance", {}).get("score", 0)))
            groundedness_scores.append(
                float(scores.get("groundedness", {}).get("score", 0))
            )
            retrieval_relevance_scores.append(
                float(scores.get("retrieval_relevance", {}).get("score", 0))
            )
            overall_scores.append(float(scores.get("avg_score", 0)))

        count = len(details) if details else 1
        summary[strategy] = {
            "avg_relevance": round(sum(relevance_scores) / count, 1),
            "avg_groundedness": round(sum(groundedness_scores) / count, 1),
            "avg_retrieval_relevance": round(
                sum(retrieval_relevance_scores) / count,
                1,
            ),
            "avg_overall": round(sum(overall_scores) / count, 1),
        }

    return summary


def load_queries(queries_path: str) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def resolve_eval_paths(base_dir: str) -> Dict[str, str]:
    env_queries = os.getenv("HK_LEGAL_EVAL_QUERIES_PATH", "").strip()
    env_output = os.getenv("HK_LEGAL_EVAL_OUTPUT_PATH", "").strip()

    queries_candidates = [
        env_queries,
        os.path.join(base_dir, "data", "queries.jsonl"),
        os.path.join(base_dir, "tests", "data", "queries.jsonl"),
        os.path.join(os.path.dirname(base_dir), "backend", "data", "queries.jsonl"),
    ]

    resolved_queries = ""
    for candidate in queries_candidates:
        if candidate and os.path.exists(candidate):
            resolved_queries = candidate
            break

    if not resolved_queries:
        searched = "\n".join(f"- {p}" for p in queries_candidates if p)
        raise FileNotFoundError(
            "Could not find queries.jsonl. Checked:\n"
            f"{searched}\n"
            "Set HK_LEGAL_EVAL_QUERIES_PATH to an absolute path if your layout differs."
        )

    resolved_output = (
        env_output
        if env_output
        else os.path.join(base_dir, "data", "eval_results.json")
    )

    output_dir = os.path.dirname(resolved_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    return {
        "queries_path": resolved_queries,
        "output_path": resolved_output,
    }


def configure_embedding_runtime_for_eval() -> None:
    env_value = os.getenv("EMBEDDING_REQUIRE_TENSORRT")
    if env_value is not None:
        return

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        providers = []

    if "TensorrtExecutionProvider" not in providers:
        os.environ["EMBEDDING_REQUIRE_TENSORRT"] = "0"
        print(
            "[Eval] EMBEDDING_REQUIRE_TENSORRT not set. "
            "Defaulting to CPU fallback because TensorRT EP is unavailable."
        )


def configure_precision_mode_for_eval() -> None:
    env_value = os.getenv("EMBEDDING_STRICT_FP16")
    if env_value is not None:
        return

    os.environ["EMBEDDING_STRICT_FP16"] = "0"
    print(
        "[Eval] EMBEDDING_STRICT_FP16 not set. "
        "Defaulting to relaxed mode for mixed-precision/legacy index compatibility."
    )


def configure_reranker_for_eval() -> None:
    env_value = os.getenv("ENABLE_RERANKER")
    if env_value is not None:
        return

    os.environ["ENABLE_RERANKER"] = "0"
    print("[Eval] ENABLE_RERANKER not set. Defaulting to reranker-disabled mode.")


async def run_evaluation() -> None:
    vector_top_k = 10
    rerank_top_k = 5
    model_name = "deepseek-chat"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_paths = resolve_eval_paths(base_dir)
    queries_path = resolved_paths["queries_path"]
    output_path = resolved_paths["output_path"]

    current_executable = _safe_str(sys.executable, "")
    print(f"[Eval] Python executable: {current_executable}")
    if "/usr/bin\\python.exe" in current_executable:
        print(
            "[Eval] WARNING: Detected mixed Linux/Windows python path. "
            "Use one environment consistently (WSL python3 or Windows .venv interpreter)."
        )

    queries = load_queries(queries_path)
    num_queries = len(queries)
    output_json_stdout = os.getenv(
        "HK_LEGAL_EVAL_STDOUT_JSON", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    print(f"[Eval] Loaded {num_queries} queries from {queries_path}")

    configure_embedding_runtime_for_eval()
    configure_precision_mode_for_eval()
    configure_reranker_for_eval()

    vs_manager = VectorStoreManager()
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    details: List[Dict[str, Any]] = []

    strategies = [
        "plain_vector",
        "rewritten_vector",
        "rewritten_expanded",
        "rewritten_rerank_expanded",
    ]
    total_steps = num_queries * len(strategies)

    with tqdm(
        total=total_steps, desc="Evaluation progress", unit="strategy"
    ) as progress:
        for idx, item in enumerate(queries, start=1):
            query = str(item.get("query", "")).strip()
            query_id = str(item.get("id", idx))
            print(f'[Eval] Query {idx}/{num_queries}: "{query}"')

            rewritten_query: Any = None
            try:
                rewritten_query = await rewrite_query(query, llm)
            except Exception as exc:
                print(f"[Eval]   Rewrite failed for query_id={query_id}: {exc}")
                rewritten_query = query

            query_result: Dict[str, Any] = {
                "query_id": query_id,
                "query": query,
                "rewritten_query": rewritten_query,
                "strategies": {},
            }

            for strategy in strategies:
                progress.set_postfix({"query": query_id, "strategy": strategy})
                try:
                    if strategy == "plain_vector":
                        results = vs_manager.search(query, k=vector_top_k)
                        sections = collapse_documents_to_sections(results)
                    elif strategy == "rewritten_vector":
                        results = vs_manager.search(rewritten_query, k=vector_top_k)
                        sections = collapse_documents_to_sections(results)
                    elif strategy == "rewritten_expanded":
                        results = vs_manager.search_with_expansion(
                            rewritten_query,
                            k=vector_top_k,
                        )
                        sections = collapse_expanded_sections(results)
                    else:
                        results = vs_manager.search_with_rerank_and_expansion(
                            rewritten_query,
                            k=vector_top_k,
                            rerank_top_k=rerank_top_k,
                        )
                        sections = collapse_expanded_sections(results)

                    context_text = build_context_text_from_sections(sections)
                    num_sources = len(sections)

                    answer = await generate_answer(query, context_text, llm)
                    scores = await evaluate_strategy(
                        strategy,
                        query,
                        context_text,
                        answer,
                        llm,
                    )
                    source_usefulness = await assess_sources_usefulness(
                        query,
                        answer,
                        sections,
                        llm,
                    )

                    retrieved_sections: List[Dict[str, Any]] = []
                    for source_idx, section in enumerate(sections, start=1):
                        retrieved_sections.append(
                            {
                                "source_index": source_idx,
                                "doc_id": section.get("doc_id", "Unknown"),
                                "section_id": section.get("section_id", "Unknown"),
                                "citation": section.get("citation", "Unknown"),
                                "section_title": section.get(
                                    "section_title", "Unknown"
                                ),
                                "source_url": section.get("source_url", ""),
                                "pages": section.get("pages", []),
                                "content": section.get("content", ""),
                            }
                        )

                    strategy_result = {
                        "answer": answer,
                        "num_sources": num_sources,
                        "retrieved_sections": retrieved_sections,
                        "source_usefulness": source_usefulness,
                        "scores": scores,
                    }
                    query_result["strategies"][strategy] = strategy_result
                    useful_count = sum(
                        1
                        for source in source_usefulness
                        if source.get("is_useful", False)
                    )
                    print(
                        f"[Eval]   Strategy: {strategy} — avg: {scores['avg_score']:.1f} "
                        f"| useful sections: {useful_count}/{num_sources}"
                    )
                    for source in source_usefulness:
                        print(
                            "[Eval]     "
                            f"S{source.get('source_index')}: "
                            f"{source.get('citation', 'Unknown')} | "
                            f"useful={source.get('is_useful', False)} | "
                            f"score={float(source.get('usefulness_score', 0)):.1f} | "
                            f"reason={source.get('reasoning', '')}"
                        )
                except Exception as exc:
                    err = f"Strategy error ({strategy}) for query_id={query_id}: {exc}"
                    print(f"[Eval]   {err}")
                    query_result["strategies"][strategy] = make_failed_strategy_result(
                        err
                    )
                finally:
                    progress.update(1)

            details.append(query_result)

    summary = compute_strategy_summary(details)
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "vector_top_k": vector_top_k,
            "rerank_top_k": rerank_top_k,
            "llm_model": model_name,
            "num_queries": num_queries,
        },
        "summary": summary,
        "details": details,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    if output_json_stdout:
        print("\n[Eval] JSON output")
        print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n[Eval] Summary")
    print(
        "Strategy                         "
        "| Relevance | Groundedness | RetrievalRel | Overall"
    )
    print("-" * 80)
    for strategy_name, scores in summary.items():
        print(
            f"{strategy_name:<32} | "
            f"{scores['avg_relevance']:>9.1f} | "
            f"{scores['avg_groundedness']:>11.1f} | "
            f"{scores['avg_retrieval_relevance']:>12.1f} | "
            f"{scores['avg_overall']:>7.1f}"
        )
    print(f"\n[Eval] Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
