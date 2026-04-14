import os
import sys
import asyncio
import json
import time
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import SecretStr

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env
from backend.core.ollama_runtime import (
    post_ollama_chat_with_fallback,
    verify_ollama_ready,
)

setup_env.setup_cuda_dlls()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.vector_store import VectorStoreManager
from backend.core.utils import (
    rewrite_query,
    generate_multi_hyde_passages,
)

load_dotenv()
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
)


async def verify_ollama_connectivity(model_name: str, ollama_base_url: str) -> None:
    await verify_ollama_ready(ollama_base_url)
    probe_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 1,
            "num_ctx": 128000,
            "num_predict": 1,
        },
    }
    await post_ollama_chat_with_fallback(probe_payload, ollama_base_url)


async def generate_ollama_answer(
    query: str,
    context_text: str,
    model_name: str,
    ollama_base_url: str,
) -> str:
    system_prompt = (
        "You are an expert Hong Kong legal assistant with comprehensive knowledge of Hong Kong ordinances and regulations. "
        "Answer the user's question based ONLY on the provided legal context. "
        "If the context doesn't contain enough information, say so. "
        "Always cite the specific Section and Ordinance when possible."
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt + "\n\nCONTEXT:\n" + context_text,
            },
            {"role": "user", "content": query},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 1,
            "num_ctx": 128000,
            "num_predict": 2048,
        },
    }

    data = await post_ollama_chat_with_fallback(payload, ollama_base_url)
    message = data.get("message", {})
    return str(message.get("content", "")).strip()


async def generate_ollama_answer_no_rag(
    query: str,
    model_name: str,
    ollama_base_url: str,
) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Please provide legal advice for the following question about Hong Kong law:\n\n"
                    f"{query}"
                ),
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 1,
            "num_ctx": 128000,
            "num_predict": 2048,
        },
    }

    data = await post_ollama_chat_with_fallback(payload, ollama_base_url)
    message = data.get("message", {})
    return str(message.get("content", "")).strip()


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
        score = max(0.0, min(10.0, score))
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
        "You are an expert Hong Kong legal assistant with comprehensive knowledge of Hong Kong ordinances and regulations. "
        "Answer the user's question based ONLY on the provided legal context. "
        "If the context doesn't contain enough information, say so. "
        "Always cite the specific Section and Ordinance when possible."
    )

    messages = [
        SystemMessage(content=system_prompt + "\n\nCONTEXT:\n" + context_text),
        HumanMessage(content=query),
    ]
    response = await llm.ainvoke(messages)
    return str(response.content).strip()


async def judge_relevance(query: str, answer: str, llm: ChatOpenAI) -> Dict[str, Any]:
    system_prompt = """You are an impartial evaluator.
Task: Evaluate how relevant and helpful the assistant's response is to the user's query.

Scoring Criteria (0.0–10.0):
- 0–2: Completely irrelevant or nonsensical response
- 3–4: Mostly irrelevant, only minor overlap with query intent
- 5–6: Partially relevant but misses key aspects of the query
- 7–8: Mostly relevant, addresses the main intent but may lack depth or completeness
- 9–10: Highly relevant, directly and fully answers the query with clarity and completeness

Evaluation Guidelines:
- Focus on whether the response answers the actual question asked
- Ignore factual correctness unless it affects relevance
- Penalize missing key components of the query
- Reward completeness, clarity, and directness
"""
    user_prompt = (
        f"User Query: {query}\n\n"
        f"Assistant Response: {answer}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
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
    system_prompt = """You are an impartial evaluator.
Task: Evaluate how well the assistant's response is grounded in the provided retrieved documents.

Scoring Criteria (0.0–10.0):
- 0–2: Response contradicts context or is entirely unsupported
- 3–4: Major parts are unsupported or hallucinated
- 5–6: Some claims are supported, but others are ungrounded or unclear
- 7–8: Mostly grounded, minor unsupported details
- 9–10: Fully grounded, all claims traceable to the context

Evaluation Guidelines:
- Check whether each claim in the response is supported by the context
- Penalize hallucinated facts or unsupported details
- Do NOT penalize missing information unless it introduces fabrication
- If the response includes external knowledge not present in context, penalize accordingly
"""
    user_prompt = (
        f"Retrieved Context: {context_text}\n\n"
        f"Assistant Response: {answer}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
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
    system_prompt = """You are an impartial evaluator.
Task: Evaluate how relevant the retrieved documents are for answering the user's query.

Scoring Criteria (0.0–10.0):
- 0–2: Completely irrelevant documents
- 3–4: Mostly irrelevant, minimal useful information
- 5–6: Some relevant content with noticeable noise, but still materially useful
- 7–8: Mostly relevant with enough legal basis to support a good answer
- 9–10: Highly relevant, directly contains the key information needed to answer the query

Evaluation Guidelines:
- Focus only on the usefulness of the retrieved documents for answering the query
- Do NOT evaluate the assistant's response here
- Penalize noise, redundancy, or missing key information proportionally (avoid all-or-nothing penalties)
- Reward retrieval that contains at least one strong legal anchor (e.g., relevant ordinance/section),
  even when some additional context is mixed in
- Reward retrieval that provides comprehensive coverage of the query's legal aspects, even if not perfectly focused, 
  and you think you can extract a good answer from it
"""
    user_prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Context: {context_text}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}


async def generate_answer_no_rag(query: str, llm: ChatOpenAI) -> str:
    messages = [
        HumanMessage(
            content=f"Please provide legal advice for the following question about Hong Kong law:\n\n{query}"
        ),
    ]
    response = await llm.ainvoke(messages)
    return str(response.content).strip()


async def judge_helpfulness(query: str, answer: str, llm: ChatOpenAI) -> Dict[str, Any]:
    system_prompt = (
        "You are an impartial evaluator.\n"
        "Task: Evaluate how helpful and actionable the assistant's legal advice is.\n\n"
        "Scoring Criteria (0.0-10.0):\n"
        "- 0-2: Completely unhelpful, no useful information\n"
        "- 3-4: Minimally helpful, vague or generic advice\n"
        "- 5-6: Somewhat helpful, provides some guidance but lacks specificity\n"
        "- 7-8: Helpful, provides actionable advice with reasonable specificity\n"
        "- 9-10: Highly helpful, comprehensive, actionable, and well-structured advice\n\n"
        "Guidelines:\n"
        "- Focus on practical usefulness to someone seeking legal guidance\n"
        "- Reward specific, actionable steps or explanations\n"
        "- Reward proper legal context and caveats\n"
        "- Penalize vague, generic, or misleading advice"
    )
    user_prompt = (
        f"User Query: {query}\n\n"
        f"Assistant Response: {answer}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}


async def judge_groundedness_vs_docs(
    answer: str, reference_context: str, llm: ChatOpenAI
) -> Dict[str, Any]:
    system_prompt = (
        "You are an impartial evaluator.\n"
        "Task: Evaluate how well the assistant's response (generated WITHOUT access to legal documents) "
        "is grounded in the actual legal documents provided as reference.\n"
        "This measures hallucination.\n\n"
        "Scoring Criteria (0.0-10.0):\n"
        "- 0-2: Response contains mostly fabricated or incorrect legal information\n"
        "- 3-4: Major legal claims are unsupported or incorrect\n"
        "- 5-6: Some claims align with documents, but significant unsupported assertions\n"
        "- 7-8: Most claims are supported, minor inaccuracies\n"
        "- 9-10: All claims are verifiable against the reference documents\n\n"
        "Guidelines:\n"
        "- Compare each legal claim against the reference documents\n"
        "- Penalize fabricated section numbers, case references, or legal principles\n"
        "- Penalize incorrect legal conclusions not supported by documents\n"
        "- Reward accurate general principles even if specific citations are missing"
    )
    user_prompt = (
        f"Reference Legal Documents (ground truth):\n{reference_context}\n\n"
        f"Assistant Response (generated WITHOUT documents):\n{answer}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}


async def evaluate_no_rag_baseline(
    query: str,
    no_rag_answer: str,
    reference_context: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    async def _limited_call(coro):
        async with semaphore:
            return await coro

    relevance, helpfulness, groundedness_vs_docs = await asyncio.wait_for(
        asyncio.gather(
            _limited_call(judge_relevance(query, no_rag_answer, llm)),
            _limited_call(judge_helpfulness(query, no_rag_answer, llm)),
            _limited_call(
                judge_groundedness_vs_docs(no_rag_answer, reference_context, llm)
            ),
        ),
        timeout=timeout_seconds,
    )
    avg_score = (
        relevance["score"] + helpfulness["score"] + groundedness_vs_docs["score"]
    ) / 3
    return {
        "relevance": relevance,
        "helpfulness": helpfulness,
        "groundedness_vs_docs": groundedness_vs_docs,
        "avg_score": round(avg_score, 1),
    }


async def evaluate_strategy(
    strategy_name: str,
    query: str,
    context_text: str,
    answer: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    rewritten_query: str = "",
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    async def _limited_call(coro):
        async with semaphore:
            return await coro

    relevance, groundedness, retrieval_relevance = await asyncio.wait_for(
        asyncio.gather(
            _limited_call(judge_relevance(query, answer, llm)),
            _limited_call(judge_groundedness(answer, context_text, llm)),
            _limited_call(
                judge_retrieval_relevance(rewritten_query or query, context_text, llm)
            ),
        ),
        timeout=timeout_seconds,
    )
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


def make_partial_strategy_result(
    error_message: str,
    sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    failed_score = {"score": 0, "reasoning": error_message}
    serialized_sections = _serialize_retrieved_sections(sections)
    return {
        "answer": "",
        "num_sources": len(sections),
        "retrieved_sections": serialized_sections,
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
) -> Dict[str, Dict[str, Any]]:
    strategy_names = set()
    for item in details:
        strategy_names.update(item.get("strategies", {}).keys())
    strategy_names = sorted(strategy_names)
    summary: Dict[str, Dict[str, Any]] = {}

    for strategy in strategy_names:
        relevance_scores: List[float] = []
        groundedness_scores: List[float] = []
        retrieval_relevance_scores: List[float] = []
        helpfulness_scores = []
        groundedness_vs_docs_scores = []
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
            helpfulness_scores.append(
                float(scores.get("helpfulness", {}).get("score", 0))
            )
            groundedness_vs_docs_scores.append(
                float(scores.get("groundedness_vs_docs", {}).get("score", 0))
            )
            overall_scores.append(float(scores.get("avg_score", 0)))

        count = len(details) if details else 1
        strat_summary = {
            "avg_relevance": round(sum(relevance_scores) / count, 1),
            "avg_groundedness": round(sum(groundedness_scores) / count, 1),
            "avg_retrieval_relevance": round(
                sum(retrieval_relevance_scores) / count,
                1,
            ),
            "avg_overall": round(sum(overall_scores) / count, 1),
        }
        if any(s > 0 for s in helpfulness_scores):
            strat_summary["avg_helpfulness"] = round(sum(helpfulness_scores) / count, 1)
        if any(s > 0 for s in groundedness_vs_docs_scores):
            strat_summary["avg_groundedness_vs_docs"] = round(
                sum(groundedness_vs_docs_scores) / count, 1
            )
        summary[strategy] = strat_summary

    baseline = summary.get("no_rag_baseline", {})
    baseline_groundedness = float(baseline.get("avg_groundedness_vs_docs", 0.0))

    for strategy, strat_summary in summary.items():
        if strategy == "no_rag_baseline":
            strat_summary["increased_performance"] = {
                "role": "baseline",
                "reference_metric": "avg_groundedness_vs_docs",
                "reference_score": round(baseline_groundedness, 1),
            }
            continue

        target_retrieval_relevance = float(
            strat_summary.get("avg_retrieval_relevance", 0.0)
        )
        absolute_delta = round(
            target_retrieval_relevance - baseline_groundedness,
            1,
        )
        increase_percentage = None
        if baseline_groundedness > 0:
            increase_percentage = round(
                (
                    (target_retrieval_relevance - baseline_groundedness)
                    / baseline_groundedness
                )
                * 100,
                1,
            )

        strat_summary["increased_performance"] = {
            "reference_strategy": "no_rag_baseline",
            "reference_metric": "avg_groundedness_vs_docs",
            "target_metric": "avg_retrieval_relevance",
            "reference_score": round(baseline_groundedness, 1),
            "target_score": round(target_retrieval_relevance, 1),
            "absolute_delta": absolute_delta,
            "increase_percentage": increase_percentage,
            "is_comparable": baseline_groundedness > 0,
            "interpretation": (
                "Proxy metric only: compares source helpfulness signal against "
                "no-RAG grounding baseline; not a direct hallucination-rate metric."
            ),
        }

    return summary


def compute_scenario_summary(
    details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for item in details:
        scenario = _safe_str(item.get("scenario", ""), "").strip() or "Unknown"
        if scenario not in grouped:
            grouped[scenario] = []
        grouped[scenario].append(item)

    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scenario in sorted(grouped.keys()):
        summary[scenario] = compute_strategy_summary(grouped[scenario])

    return summary


def compute_retrieval_summary(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_queries = len(details)
    if num_queries == 0:
        return {
            "mode": "retrieval_only",
            "num_queries": 0,
            "avg_num_sources": 0.0,
            "avg_unique_docs": 0.0,
        }

    source_counts: List[int] = []
    unique_doc_counts: List[int] = []

    preferred_strategy = "multi_hyde_reranked"

    for item in details:
        strategies_map = item.get("strategies", {})
        strategy_data = strategies_map.get(preferred_strategy, {})
        retrieved_sections = strategy_data.get("retrieved_sections", [])
        source_counts.append(int(strategy_data.get("num_sources", 0)))

        unique_docs = {
            str(section.get("doc_id", "Unknown"))
            for section in retrieved_sections
            if isinstance(section, dict)
        }
        unique_doc_counts.append(len(unique_docs))

    return {
        "mode": "retrieval_only",
        "num_queries": num_queries,
        "avg_num_sources": round(sum(source_counts) / num_queries, 2),
        "avg_unique_docs": round(sum(unique_doc_counts) / num_queries, 2),
    }


def _extract_page_anchor_link(section: Dict[str, Any]) -> str:
    source_url = _safe_str(section.get("source_url", ""), "").strip()
    if not source_url:
        return ""

    pages = section.get("pages", [])
    if isinstance(pages, list) and pages:
        page_text = _safe_str(pages[0], "").strip()
        if page_text.isdigit():
            separator = "&" if "#" in source_url else "#"
            return f"{source_url}{separator}page={page_text}"

    return source_url


def _tokenize_for_overlap(text: str) -> set[str]:
    text = _safe_str(text, "")
    latin_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    non_ascii_chars = [ch for ch in text if ord(ch) > 127 and not ch.isspace()]

    token_set = {token for token in latin_tokens if len(token) > 2}
    token_set.update(non_ascii_chars)
    return token_set


def compute_retrieval_only_scores(
    query: str,
    rewritten_query: str,
    sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not sections:
        empty_score = {
            "score": 0.0,
            "reasoning": "No sections retrieved from vector search.",
        }
        return {
            "relevance": empty_score,
            "groundedness": empty_score,
            "retrieval_relevance": empty_score,
            "avg_score": 0.0,
        }

    query_tokens = _tokenize_for_overlap(query)
    rewritten_tokens = _tokenize_for_overlap(rewritten_query)
    target_tokens = query_tokens.union(rewritten_tokens)

    coverage_scores: List[float] = []
    total_chars = 0
    unique_doc_ids = set()

    for section in sections:
        content = _safe_str(section.get("content", ""), "")
        total_chars += len(content)
        unique_doc_ids.add(_safe_str(section.get("doc_id", "Unknown"), "Unknown"))

        content_tokens = _tokenize_for_overlap(content)
        if not target_tokens:
            overlap_ratio = 1.0
        else:
            overlap_ratio = len(target_tokens.intersection(content_tokens)) / max(
                1, len(target_tokens)
            )
        coverage_scores.append(overlap_ratio)

    avg_coverage = sum(coverage_scores) / max(1, len(coverage_scores))
    max_coverage = max(coverage_scores) if coverage_scores else 0.0
    source_count = len(sections)
    unique_doc_count = len(unique_doc_ids)

    relevance_raw = max_coverage * 10.0
    retrieval_relevance_raw = avg_coverage * 10.0

    if sections:
        relevance_raw = max(0.5, relevance_raw)
        retrieval_relevance_raw = max(0.5, retrieval_relevance_raw)

    relevance = round(min(10.0, relevance_raw), 1)
    retrieval_relevance = round(min(10.0, retrieval_relevance_raw), 1)

    richness_ratio = min(1.0, total_chars / 6000.0)
    diversity_ratio = min(1.0, unique_doc_count / max(1, source_count))
    groundedness_raw = (0.7 * richness_ratio) + (0.3 * diversity_ratio)
    groundedness = round(min(10.0, groundedness_raw * 10.0), 1)

    avg_score = round((relevance + groundedness + retrieval_relevance) / 3.0, 1)

    return {
        "relevance": {
            "score": relevance,
            "reasoning": (
                "Estimated from max query-token overlap with retrieved section content "
                f"(max_overlap={max_coverage:.2f})."
            ),
        },
        "groundedness": {
            "score": groundedness,
            "reasoning": (
                "Estimated from retrieved context richness and document diversity "
                f"(chars={total_chars}, unique_docs={unique_doc_count})."
            ),
        },
        "retrieval_relevance": {
            "score": retrieval_relevance,
            "reasoning": (
                "Estimated from average query-token overlap across retrieved sections "
                f"(avg_overlap={avg_coverage:.2f})."
            ),
        },
        "avg_score": avg_score,
    }


def _collect_strategy_score_row(scores: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "relevance": scores.get("relevance", {}).get("score", 0),
        "groundedness": scores.get("groundedness", {}).get("score", 0),
        "retrieval_relevance": scores.get("retrieval_relevance", {}).get("score", 0),
        "avg_score": scores.get("avg_score", 0),
    }
    if "helpfulness" in scores:
        row["helpfulness"] = scores["helpfulness"].get("score", 0)
    if "groundedness_vs_docs" in scores:
        row["groundedness_vs_docs"] = scores["groundedness_vs_docs"].get("score", 0)
    return row


def build_score_report(
    eval_mode: str,
    timestamp: str,
    details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    query_scores: List[Dict[str, Any]] = []

    for item in details:
        strategies = item.get("strategies", {})
        strategy_scores: Dict[str, Dict[str, Any]] = {}
        for strategy_name, strategy_data in strategies.items():
            strategy_scores[strategy_name] = _collect_strategy_score_row(
                strategy_data.get("scores", {})
            )

        query_scores.append(
            {
                "query_id": item.get("query_id", ""),
                "query": item.get("query", ""),
                "rewritten_query": item.get("rewritten_query", ""),
                "scenario": item.get("scenario", ""),
                "scores_by_strategy": strategy_scores,
            }
        )

    return {
        "timestamp": timestamp,
        "general_summary": {
            "by_scenario": compute_scenario_summary(details),
            "by_strategy": compute_strategy_summary(details),
        },
        "queries": query_scores,
    }


def _collect_strategy_source_links(
    retrieved_sections: List[Dict[str, Any]],
) -> List[str]:
    links: List[str] = []
    seen = set()

    for section in retrieved_sections:
        if not isinstance(section, dict):
            continue
        link = _extract_page_anchor_link(section)
        if not link or link in seen:
            continue
        seen.add(link)
        links.append(link)

    return links


def build_sources_report(
    eval_mode: str,
    timestamp: str,
    details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_query: List[Dict[str, Any]] = []
    unique_links = set()

    for item in details:
        strategy_sources: Dict[str, Dict[str, Any]] = {}
        for strategy_name, strategy_data in item.get("strategies", {}).items():
            retrieved_sections = strategy_data.get("retrieved_sections", [])
            source_links = _collect_strategy_source_links(retrieved_sections)
            unique_links.update(source_links)

            source_entry: Dict[str, Any] = {
                "source_links": source_links,
                "num_sources": int(strategy_data.get("num_sources", 0)),
            }

            strategy_sources[strategy_name] = source_entry

        by_query.append(
            {
                "query_id": item.get("query_id", ""),
                "query": item.get("query", ""),
                "rewritten_query": item.get("rewritten_query", ""),
                "scenario": item.get("scenario", ""),
                "strategies": strategy_sources,
            }
        )

    general_explanation = (
        "Source links are grouped per query and per retrieval strategy so you can compare "
        "which ordinance sections each strategy surfaced. "
        f"Across {len(details)} queries, the report captured {len(unique_links)} unique source links."
    )

    return {
        "timestamp": timestamp,
        "by_query": by_query,
        "general_explanation": general_explanation,
    }


def load_queries(queries_path: str) -> List[Dict[str, Any]]:
    with open(queries_path, "r", encoding="utf-8") as file:
        raw = file.read().strip()
    if not raw:
        return []
    # JSON array (e.g. better_queries.jsonl which is actually JSON)
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(
                f"Expected JSON array in {queries_path}, got {type(parsed).__name__}"
            )
        return parsed
    # JSONL (one JSON object per line)
    queries: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        queries.append(json.loads(line))
    return queries


def filter_queries_by_id_pattern(
    queries: List[Dict[str, Any]],
    pattern: str,
) -> List[Dict[str, Any]]:
    compiled = re.compile(pattern)
    filtered: List[Dict[str, Any]] = []
    for item in queries:
        query_id = _safe_str(item.get("id", ""), "")
        if compiled.fullmatch(query_id):
            filtered.append(item)
    return filtered


def resolve_eval_paths(base_dir: str) -> Dict[str, str]:
    env_output = os.getenv("HK_LEGAL_EVAL_OUTPUT_PATH", "").strip()

    output_dir = env_output if env_output else os.path.join(base_dir, "data")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    return {"output_dir": output_dir}


def resolve_query_file_paths(base_dir: str) -> List[str]:
    single_queries = os.getenv("HK_LEGAL_EVAL_QUERIES_PATH", "").strip()
    multi_queries_raw = os.getenv("HK_LEGAL_EVAL_QUERIES_PATHS", "").strip()

    if single_queries:
        requested_paths = [single_queries]
    elif multi_queries_raw:
        requested_paths = [
            part.strip() for part in multi_queries_raw.split(",") if part.strip()
        ]
    else:
        requested_paths = [
            os.path.join(base_dir, "data", "queries.jsonl"),
            os.path.join(base_dir, "data", "better_queries.jsonl"),
        ]

    repo_root = os.path.dirname(base_dir)
    resolved_paths: List[str] = []
    for path in requested_paths:
        candidates = [path]
        if not os.path.isabs(path):
            candidates.append(os.path.join(base_dir, path))
            candidates.append(os.path.join(repo_root, path))

        resolved = ""
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved = candidate
                break

        if not resolved:
            searched = "\n".join(f"- {candidate}" for candidate in candidates)
            raise FileNotFoundError(
                "Could not find query file. Checked:\n"
                f"{searched}\n"
                "Set HK_LEGAL_EVAL_QUERIES_PATHS to valid paths, or use HK_LEGAL_EVAL_QUERIES_PATH."
            )

        resolved_paths.append(resolved)

    return resolved_paths


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


def configure_reranker_runtime_for_eval() -> None:
    force_cpu = os.getenv("HK_LEGAL_EVAL_RERANKER_FORCE_CPU", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if force_cpu:
        os.environ["RERANKER_FORCE_CPU"] = "1"
        os.environ["RERANKER_REQUIRE_TENSORRT"] = "0"
        print(
            "[Eval] Reranker runtime configured for CPU (RERANKER_FORCE_CPU=1, "
            "RERANKER_REQUIRE_TENSORRT=0)."
        )
    else:
        require_trt = os.getenv(
            "HK_LEGAL_EVAL_RERANKER_REQUIRE_TENSORRT", "0"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        os.environ["RERANKER_FORCE_CPU"] = "0"
        os.environ["RERANKER_REQUIRE_TENSORRT"] = "1" if require_trt else "0"
        print(
            "[Eval] Reranker runtime configured for TRT attempt "
            f"(RERANKER_FORCE_CPU=0, RERANKER_REQUIRE_TENSORRT={os.environ['RERANKER_REQUIRE_TENSORRT']})."
        )


def configure_trt_workspace_for_eval() -> None:
    eval_embedding_workspace = int(
        os.getenv("HK_LEGAL_EVAL_EMBEDDING_TRT_MAX_WORKSPACE_SIZE", str(3 * 1024**3))
    )
    eval_reranker_workspace = int(
        os.getenv("HK_LEGAL_EVAL_RERANKER_TRT_MAX_WORKSPACE_SIZE", str(1 * 1024**3))
    )

    os.environ["EMBEDDING_TRT_MAX_WORKSPACE_SIZE"] = str(eval_embedding_workspace)
    os.environ["RERANKER_TRT_MAX_WORKSPACE_SIZE"] = str(eval_reranker_workspace)

    print(
        "[Eval] TRT workspaces configured: "
        f"EMBEDDING_TRT_MAX_WORKSPACE_SIZE={eval_embedding_workspace}, "
        f"RERANKER_TRT_MAX_WORKSPACE_SIZE={eval_reranker_workspace}."
    )


async def _run_with_semaphore(semaphore: asyncio.Semaphore, coro):
    async with semaphore:
        return await coro


async def _run_with_timeout(coro, timeout_seconds: int, label: str):
    if timeout_seconds <= 0:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout_seconds}s") from exc


async def _run_blocking_with_timeout(
    func, timeout_seconds: int, label: str, *args, **kwargs
):
    if timeout_seconds <= 0:
        return await asyncio.to_thread(func, *args, **kwargs)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout_seconds}s") from exc


def _serialize_retrieved_sections(
    sections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    retrieved_sections: List[Dict[str, Any]] = []
    for source_idx, section in enumerate(sections, start=1):
        retrieved_sections.append(
            {
                "source_index": source_idx,
                "doc_id": section.get("doc_id", "Unknown"),
                "section_id": section.get("section_id", "Unknown"),
                "citation": section.get("citation", "Unknown"),
                "section_title": section.get("section_title", "Unknown"),
                "source_url": section.get("source_url", ""),
                "pages": section.get("pages", []),
                "content": section.get("content", ""),
            }
        )
    return retrieved_sections


async def run_evaluation() -> None:
    vector_top_k = int(os.getenv("HK_LEGAL_EVAL_VECTOR_TOP_K", "50"))
    rerank_top_k = int(os.getenv("HK_LEGAL_EVAL_RERANK_TOP_K", "10"))
    model_name = os.getenv("HK_LEGAL_EVAL_MODEL", "qwen3.5:9b")
    judge_model_name = os.getenv("HK_LEGAL_EVAL_JUDGE_MODEL", "deepseek-chat")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    eval_mode = "full"
    progress_only = os.getenv("HK_LEGAL_EVAL_PROGRESS_ONLY", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    max_concurrent_llm = int(os.getenv("HK_LEGAL_EVAL_LLM_CONCURRENCY", "6"))
    hyde_timeout_seconds = int(os.getenv("HK_LEGAL_EVAL_HYDE_TIMEOUT_SECONDS", "180"))
    retrieval_timeout_seconds = int(
        os.getenv("HK_LEGAL_EVAL_RETRIEVAL_TIMEOUT_SECONDS", "180")
    )
    answer_timeout_seconds = int(
        os.getenv("HK_LEGAL_EVAL_ANSWER_TIMEOUT_SECONDS", "300")
    )
    judge_timeout_seconds = int(os.getenv("HK_LEGAL_EVAL_JUDGE_TIMEOUT_SECONDS", "300"))
    skip_source_usefulness = os.getenv(
        "HK_LEGAL_EVAL_SKIP_SOURCE_USEFULNESS", "1"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    multi_hyde_k_per_query = max(
        1, int(os.getenv("HK_LEGAL_EVAL_MULTI_HYDE_K_PER_QUERY", "3"))
    )
    multi_hyde_rerank_top_k = max(
        1, int(os.getenv("HK_LEGAL_EVAL_MULTI_HYDE_RERANK_TOP_K", "5"))
    )
    semaphore = asyncio.Semaphore(max_concurrent_llm)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_paths = resolve_eval_paths(base_dir)
    output_dir = resolved_paths["output_dir"]
    query_files = resolve_query_file_paths(base_dir)

    current_executable = _safe_str(sys.executable, "")
    if not progress_only:
        print(f"[Eval] Python executable: {current_executable}")

    if "/usr/bin\\python.exe" in current_executable and not progress_only:
        print(
            "[Eval] WARNING: Detected mixed Linux/Windows python path. "
            "Use one environment consistently (WSL python3 or Windows .venv interpreter)."
        )

    # Per-file default patterns:
    #   queries.jsonl  → type-1 only (q1_1, q2_1, …)
    #   everything else → all queries
    _DEFAULT_PATTERNS: Dict[str, str] = {
        "queries": r"q\d+_1",
    }
    query_batches: List[Tuple[str, str, str, List[Dict[str, Any]]]] = []
    output_json_stdout = os.getenv(
        "HK_LEGAL_EVAL_STDOUT_JSON", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    for query_path in query_files:
        queries = load_queries(query_path)
        suffix = os.path.splitext(os.path.basename(query_path))[0]
        pattern = _DEFAULT_PATTERNS.get(suffix, r".*")
        pre_filter_count = len(queries)
        queries = filter_queries_by_id_pattern(queries, pattern)
        query_batches.append((query_path, suffix, pattern, queries))

        print(
            f"[Eval] Loaded {pre_filter_count} raw → {len(queries)} filtered from {query_path} "
            f"(suffix={suffix!r}, pattern={pattern!r})"
        )

    total_queries = sum(len(batch_queries) for _, _, _, batch_queries in query_batches)
    if total_queries == 0:
        raise ValueError(
            "No queries selected for evaluation. "
            f"Per-file default patterns: {_DEFAULT_PATTERNS}."
        )

    configure_embedding_runtime_for_eval()
    configure_precision_mode_for_eval()
    configure_trt_workspace_for_eval()
    configure_reranker_runtime_for_eval()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for evaluation")

    judge_llm = ChatOpenAI(
        model=judge_model_name,
        api_key=SecretStr(deepseek_api_key),
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    await verify_ollama_connectivity(model_name, ollama_base_url)

    vs_manager = VectorStoreManager()

    from backend.services.embedding_service import get_embedding_service

    embedding_service = get_embedding_service()

    embedding_service.ensure_loaded()

    strategies: List[str] = ["multi_hyde_reranked", "no_rag_baseline"]
    total_steps = total_queries * len(strategies)
    saved_output_paths: List[Tuple[str, str, str, str]] = []

    try:
        with tqdm(
            total=total_steps, desc="Evaluation progress", unit="strategy"
        ) as progress:
            for (
                query_file_path,
                query_file_suffix,
                batch_pattern,
                queries,
            ) in query_batches:
                details: List[Dict[str, Any]] = []
                num_queries = len(queries)

                for idx, item in enumerate(queries, start=1):
                    query = str(item.get("query", "")).strip()
                    query_id = str(item.get("id", idx))
                    scenario = str(item.get("scenario", "")).strip()
                    if not progress_only:
                        print(
                            f'[Eval] Query {idx}/{num_queries} ({query_file_suffix}): "{query}"'
                        )

                    try:
                        rewritten_query = await _run_with_semaphore(
                            semaphore,
                            rewrite_query(query, judge_llm, scenario=scenario),
                        )
                    except Exception as exc:
                        print(f"[Eval] Rewrite failed for query_id={query_id}: {exc}")
                        rewritten_query = query

                    query_result: Dict[str, Any] = {
                        "query_id": query_id,
                        "query": query,
                        "rewritten_query": rewritten_query,
                        "scenario": scenario,
                        "strategies": {},
                    }

                    try:
                        for strategy in strategies:
                            progress.set_postfix(
                                {
                                    "file": query_file_suffix,
                                    "query": query_id,
                                    "strategy": strategy,
                                }
                            )
                            sections: List[Dict[str, Any]] = []
                            if strategy == "no_rag_baseline":
                                no_rag_answer = await _run_with_semaphore(
                                    semaphore,
                                    generate_ollama_answer_no_rag(
                                        query,
                                        model_name,
                                        ollama_base_url,
                                    ),
                                )
                                ref_strategy_data = query_result["strategies"].get(
                                    "multi_hyde_reranked", {}
                                )
                                reference_context = ""
                                if ref_strategy_data:
                                    ref_sections = ref_strategy_data.get(
                                        "retrieved_sections", []
                                    )
                                    reference_context = (
                                        build_context_text_from_sections(ref_sections)
                                    )

                                scores = await evaluate_no_rag_baseline(
                                    query,
                                    no_rag_answer,
                                    reference_context,
                                    judge_llm,
                                    semaphore,
                                    timeout_seconds=judge_timeout_seconds,
                                )

                                strategy_result = {
                                    "answer": no_rag_answer,
                                    "num_sources": 0,
                                    "retrieved_sections": [],
                                    "source_usefulness": [],
                                    "scores": scores,
                                }
                                query_result["strategies"][strategy] = strategy_result
                                progress.update(1)
                                continue
                            try:
                                if strategy == "multi_hyde_reranked":
                                    reranker_service = vs_manager.reranker
                                    if reranker_service is None:
                                        raise RuntimeError(
                                            "Reranker is disabled (ENABLE_RERANKER=0), "
                                            "cannot run multi_hyde_reranked strategy"
                                        )

                                    reranker_service.configure_model(
                                        model_dir_name=os.getenv(
                                            "RERANKER_MODEL_DIR", "zerank-1-small"
                                        ),
                                        onnx_model_file=os.getenv(
                                            "RERANKER_ONNX_FILE", "model_fp16.onnx"
                                        ),
                                    )

                                    try:
                                        if progress_only:
                                            print(
                                                f"[Eval] {query_id}:{strategy} stage=multi_hyde_generation"
                                            )
                                        multi_passages = await _run_with_semaphore(
                                            semaphore,
                                            _run_with_timeout(
                                                generate_multi_hyde_passages(
                                                    query, judge_llm
                                                ),
                                                hyde_timeout_seconds,
                                                f"Multi-HyDE generation for {query_id}",
                                            ),
                                        )

                                        if not multi_passages:
                                            raise RuntimeError(
                                                "Multi-HyDE generation returned no usable passages"
                                            )

                                        if progress_only:
                                            print(
                                                f"[Eval] {query_id}:{strategy} stage=multi_retrieval_rerank "
                                                f"passages={len(multi_passages)}"
                                            )
                                        results = await _run_blocking_with_timeout(
                                            vs_manager.search_multi_hyde_with_rerank_and_expansion,
                                            retrieval_timeout_seconds,
                                            f"Multi-HyDE retrieval+rerank for {query_id}",
                                            passages=multi_passages,
                                            original_query=query,
                                            k_per_query=multi_hyde_k_per_query,
                                            rerank_top_k=multi_hyde_rerank_top_k,
                                            capture_scores=True,
                                        )
                                        sections = collapse_expanded_sections(results)
                                    finally:
                                        if reranker_service.is_loaded():
                                            reranker_service.unload()
                                else:
                                    raise ValueError(f"Unknown strategy: {strategy}")

                                context_text = build_context_text_from_sections(
                                    sections
                                )
                                num_sources = len(sections)
                                if progress_only:
                                    print(
                                        f"[Eval] {query_id}:{strategy} stage=answer sources={num_sources}"
                                    )
                                answer = await _run_with_semaphore(
                                    semaphore,
                                    _run_with_timeout(
                                        generate_ollama_answer(
                                            query,
                                            context_text,
                                            model_name,
                                            ollama_base_url,
                                        ),
                                        answer_timeout_seconds,
                                        f"Answer generation for {query_id}",
                                    ),
                                )

                                if progress_only:
                                    print(f"[Eval] {query_id}:{strategy} stage=judge")
                                scores = await evaluate_strategy(
                                    strategy,
                                    query,
                                    rewritten_query=rewritten_query,
                                    context_text=context_text,
                                    answer=answer,
                                    llm=judge_llm,
                                    semaphore=semaphore,
                                    timeout_seconds=judge_timeout_seconds,
                                )

                                if skip_source_usefulness:
                                    source_usefulness = []
                                else:
                                    source_usefulness = await _run_with_semaphore(
                                        semaphore,
                                        assess_sources_usefulness(
                                            query,
                                            answer,
                                            sections,
                                            judge_llm,
                                        ),
                                    )

                                strategy_result: Dict[str, Any] = {
                                    "answer": answer,
                                    "num_sources": num_sources,
                                    "retrieved_sections": _serialize_retrieved_sections(
                                        sections
                                    ),
                                    "hyde_passages": multi_passages,
                                    "reranker_scores": (
                                        vs_manager._last_rerank_scores
                                        if hasattr(vs_manager, "_last_rerank_scores")
                                        else []
                                    ),
                                    "source_usefulness": source_usefulness,
                                    "scores": scores,
                                }

                                query_result["strategies"][strategy] = strategy_result

                                if not progress_only:
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
                                if not progress_only:
                                    print(f"[Eval]   {err}")

                                if sections:
                                    failed_result = make_partial_strategy_result(
                                        err, sections
                                    )
                                else:
                                    failed_result = make_failed_strategy_result(err)
                                query_result["strategies"][strategy] = failed_result
                            finally:
                                progress.update(1)
                    finally:
                        details.append(query_result)

                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                summary = {
                    "by_strategy": compute_strategy_summary(details),
                    "by_scenario": compute_scenario_summary(details),
                }

                results = {
                    "timestamp": timestamp,
                    "config": {
                        "eval_mode": eval_mode,
                        "progress_only": progress_only,
                        "query_id_pattern": batch_pattern,
                        "vector_top_k": vector_top_k,
                        "rerank_top_k": rerank_top_k,
                        "llm_model": model_name,
                        "num_queries": num_queries,
                        "max_concurrent_llm": max_concurrent_llm,
                        "skip_source_usefulness": skip_source_usefulness,
                        "multi_hyde_k_per_query": multi_hyde_k_per_query,
                        "multi_hyde_rerank_top_k": multi_hyde_rerank_top_k,
                        "query_file": query_file_path,
                    },
                    "summary": summary,
                    "details": details,
                }

                output_path = os.path.join(
                    output_dir, f"eval_results_{query_file_suffix}.json"
                )
                score_output_path = os.path.join(
                    output_dir, f"eval_scores_{query_file_suffix}.json"
                )
                sources_output_path = os.path.join(
                    output_dir, f"eval_sources_{query_file_suffix}.json"
                )

                score_report = build_score_report(
                    eval_mode=eval_mode,
                    timestamp=timestamp,
                    details=details,
                )
                sources_report = build_sources_report(
                    eval_mode=eval_mode,
                    timestamp=timestamp,
                    details=details,
                )

                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(results, file, indent=2, ensure_ascii=False)

                with open(score_output_path, "w", encoding="utf-8") as file:
                    json.dump(score_report, file, indent=2, ensure_ascii=False)

                with open(sources_output_path, "w", encoding="utf-8") as file:
                    json.dump(sources_report, file, indent=2, ensure_ascii=False)

                saved_output_paths.append(
                    (
                        query_file_suffix,
                        output_path,
                        score_output_path,
                        sources_output_path,
                    )
                )

                if output_json_stdout:
                    print("\n[Eval] JSON output")
                    print(json.dumps(results, ensure_ascii=False, indent=2))

                if not progress_only:
                    print(f"\n[Eval] Summary ({query_file_suffix})")
                    has_helpfulness = any(
                        "avg_helpfulness" in scores
                        for scores in summary["by_strategy"].values()
                    )

                    header = "Strategy                         | Relevance | Groundedness | RetrievalRel"
                    if has_helpfulness:
                        header += " | Helpfulness | Ground.Docs"
                    header += " | Overall"
                    print(header)
                    print("-" * len(header))

                    for strategy_name, scores in summary["by_strategy"].items():
                        row = (
                            f"{strategy_name:<32} | "
                            f"{scores.get('avg_relevance', 0):>9.1f} | "
                            f"{scores.get('avg_groundedness', 0):>11.1f} | "
                            f"{scores.get('avg_retrieval_relevance', 0):>12.1f}"
                        )
                        if has_helpfulness:
                            row += (
                                f" | {scores.get('avg_helpfulness', 0):>11.1f}"
                                f" | {scores.get('avg_groundedness_vs_docs', 0):>11.1f}"
                            )
                        row += f" | {scores.get('avg_overall', 0):>7.1f}"
                        print(row)
    finally:
        if embedding_service.is_loaded():
            embedding_service.unload()

    for (
        query_file_suffix,
        output_path,
        score_output_path,
        sources_output_path,
    ) in saved_output_paths:
        print(f"\n[Eval] [{query_file_suffix}] Results saved to: {output_path}")
        print(f"[Eval] [{query_file_suffix}] Scores saved to: {score_output_path}")
        print(f"[Eval] [{query_file_suffix}] Sources saved to: {sources_output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
