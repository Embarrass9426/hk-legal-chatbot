import os
import sys
import json
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env
from backend.core.utils import rewrite_query

setup_env.setup_cuda_dlls()

from backend.services.vector_store import VectorStoreManager

load_dotenv()

app = FastAPI()
llm: ChatOpenAI | None = None


def _create_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("WARNING: DEEPSEEK_API_KEY is not set correctly in .env file")

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        max_tokens=1024,
        streaming=True,
    )


def get_llm() -> ChatOpenAI:
    global llm
    if llm is None:
        llm = _create_llm()
    return llm


# Initialize components
vector_manager = VectorStoreManager()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    language: str = "en"


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def collapse_documents_to_sections(results: List[Any]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for doc in results:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_id = _safe_str(metadata.get("doc_id", "Unknown"), "Unknown")
        section_id = _safe_str(metadata.get("section_id", "Unknown"), "Unknown")
        section_title = _safe_str(
            metadata.get("section_title", "Unknown Section"),
            "Unknown Section",
        )
        citation = _safe_str(
            metadata.get("citation", f"Section {section_id}"),
            f"Section {section_id}",
        )
        source_url = _safe_str(metadata.get("source_url", ""), "")
        page_number = metadata.get("page_number")
        content = _safe_str(getattr(doc, "page_content", ""), "").strip()

        prefix = "Represent this legal document passage for retrieval: "
        if content.startswith(prefix):
            content = content[len(prefix) :]

        key = (doc_id, section_id)
        if key not in grouped:
            grouped[key] = {
                "doc_id": doc_id,
                "section_id": section_id,
                "section_title": section_title,
                "citation": citation,
                "source_url": source_url,
                "pages": set(),
                "content_parts": [],
            }

        if page_number not in (None, ""):
            grouped[key]["pages"].add(_safe_str(page_number))

        if content:
            grouped[key]["content_parts"].append(content)

    sections: List[Dict[str, Any]] = []
    for section in grouped.values():
        unique_parts: List[str] = []
        seen = set()
        for part in section["content_parts"]:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        sections.append(
            {
                "doc_id": section["doc_id"],
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "citation": section["citation"],
                "source_url": section["source_url"],
                "pages": sorted(section["pages"], key=lambda x: (len(x), x)),
                "content": "\n\n".join(unique_parts),
            }
        )

    sections.sort(key=lambda s: (s["doc_id"], s["section_id"]))
    return sections


async def assess_section_usefulness(
    query: str,
    answer: str,
    sections: List[Dict[str, Any]],
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
        "You evaluate legal source usefulness by section. "
        "For each source, return whether it is useful for answering the user question. "
        "Use usefulness_score from 0.0 to 10.0 with one decimal place."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Retrieved sections JSON:\n{json.dumps(sources_payload, ensure_ascii=False)}\n\n"
        "Return JSON only:\n"
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
        "Rules: include one object for each source_index."
    )

    try:
        response = await get_llm().ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        raw_text = _safe_str(response.content, "").strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

        parsed = json.loads(raw_text)
        source_items = parsed.get("sources", [])
        mapped = {
            int(item.get("source_index", -1)): item
            for item in source_items
            if isinstance(item, dict)
        }

        normalized: List[Dict[str, Any]] = []
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
                    "is_useful": bool(item.get("is_useful", False)),
                    "usefulness_score": numeric_score,
                    "reasoning": _safe_str(item.get("reasoning", ""), ""),
                }
            )

        return normalized
    except Exception as exc:
        fallback: List[Dict[str, Any]] = []
        for idx, _ in enumerate(sections, start=1):
            fallback.append(
                {
                    "source_index": idx,
                    "is_useful": False,
                    "usefulness_score": 0.0,
                    "reasoning": f"Source usefulness evaluation error: {exc}",
                }
            )
        return fallback


async def generate_chat_responses(message: str, language: str = "en"):
    try:
        retrieval_query = await rewrite_query(message, get_llm())
        print(f"Retrieval query: {retrieval_query}")

        search_results = vector_manager.search(retrieval_query, k=5)
        sections = collapse_documents_to_sections(search_results)

        context_parts = []
        references = []

        for i, section in enumerate(sections):
            pages = section.get("pages", [])
            pages_text = ", ".join(pages) if pages else "Unknown"
            context_parts.append(
                f"### Reference [{i + 1}]: {section.get('section_title', 'Unknown Section')}\n"
                f"Citation: {section.get('citation', 'Unknown')}\n"
                f"Pages: {pages_text}\n"
                f"{section.get('content', '')}"
            )

            references.append(
                {
                    "id": f"ref-{i}",
                    "doc_id": section.get("doc_id", "Unknown"),
                    "section_id": section.get("section_id", "Unknown"),
                    "section_title": section.get("section_title", "Unknown Section"),
                    "title": section.get("citation", "Unknown"),
                    "source_url": section.get("source_url", ""),
                    "pages": pages,
                    "jurisdiction": "Hong Kong",
                    "type": "Ordinance",
                }
            )

        context_text = "\n\n".join(context_parts)

        system_content = """You are an expert Hong Kong legal assistant specializing in the Employees' Compensation Ordinance (Cap. 282).
Your goal is to help employees understand their rights regarding workplace injuries and insurance.

Instructions:
1. Use the provided legal context to answer the user's question.
2. If the context doesn't contain the answer, state that you don't have enough information but provide general guidance based on the context.
3. Always cite the specific Section (e.g., [1] Cap. 282, s. 5) when referring to the law.
4. Be empathetic but professional.
5. If the user asks about a specific injury (like breaking a leg), explain if it's covered under "arising out of and in the course of employment".
"""

        if language == "zh":
            system_content += "\nIMPORTANT: You MUST respond in Traditional Chinese (繁體中文). Even if the context is in English, translate the relevant parts to help the user understand, but keep the legal citations (e.g. Cap. 282, s. 5) recognizable."
        else:
            system_content += "\nIMPORTANT: Respond in English."

        system_content += "\n\nCONTEXT:\n" + context_text

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=message),
        ]

        print(f"Starting stream for message: {message}")
        answer_parts: List[str] = []
        async for chunk in get_llm().astream(messages):
            if chunk.content:
                answer_parts.append(chunk.content)
                data = json.dumps({"answer": chunk.content})
                yield f"data: {data}\n\n"

        full_answer = "".join(answer_parts).strip()
        source_usefulness = await assess_section_usefulness(
            message,
            full_answer,
            sections,
        )

        references_with_usefulness: List[Dict[str, Any]] = []
        for idx, reference in enumerate(references, start=1):
            usefulness = (
                source_usefulness[idx - 1]
                if idx - 1 < len(source_usefulness)
                else {
                    "source_index": idx,
                    "is_useful": False,
                    "usefulness_score": 0.0,
                    "reasoning": "Missing usefulness evaluation.",
                }
            )
            merged = {**reference, **usefulness}
            references_with_usefulness.append(merged)

        if references_with_usefulness:
            yield f"data: {json.dumps({'references': references_with_usefulness})}\n\n"

        print("Stream finished")
    except Exception as e:
        print(f"Error in stream: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return StreamingResponse(
            generate_chat_responses(request.message, request.language),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "HK Legal Chatbot API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
