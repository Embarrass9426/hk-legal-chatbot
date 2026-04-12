import os
import re
import asyncio
import numpy as np
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage


async def rewrite_query(user_query: str, llm, scenario: str = "") -> str:
    """
    Rewrites the user question into a legal-focused retrieval query.
    Kept for backward compatibility (eval framework).
    """
    system_prompt = """You are an expert legal query rewriter for a retrieval system.
Task: Rewrite the user's query into a clear, precise, and information-rich version optimized for legal document retrieval.

Rewriting Goals:
- Preserve the original intent exactly (DO NOT change meaning)
- Expand vague or short queries into complete, explicit questions
- Add relevant legal context (e.g., workplace injury, compensation, liability, insurance, employment rights)
- Normalize informal language into clear professional wording
- Include key entities if implied (e.g., employer, insurance, compensation, medical costs)
- Clarify ambiguity (e.g., "can I claim?" → "am I eligible for compensation and reimbursement?")
- Keep it concise but complete (1–2 sentences preferred)

Special Guidance:
- If the query is very short (e.g., "Arm broken."), infer intent using the scenario
- If multiple interpretations exist, choose the most likely legal interpretation
- Do NOT introduce facts not implied by the query or scenario
- Do NOT answer the question — only rewrite it

Return ONLY the rewritten query text.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Original Query: {user_query}\nScenario: {scenario}"
            if scenario
            else f"Original Query: {user_query}"
        ),
    ]

    try:
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return user_query  # Fallback to original query


HYDE_SYSTEM_PROMPT = (
    "You are an expert Hong Kong legal assistant with comprehensive knowledge "
    "of Hong Kong ordinances and regulations.\n"
    "Your goal is to help users understand their legal rights and obligations "
    "under Hong Kong law.\n\n"
    "Instructions:\n"
    "1. Write a short passage (2-4 paragraphs) that resembles an excerpt from a "
    "Hong Kong ordinance, regulation, or official legal explanatory text that would "
    "contain the answer to the user's question.\n"
    "2. Use formal legal/statutory vocabulary.\n"
    "3. Include the likely legal test, rights, duties, conditions, exceptions, "
    "or remedies that would apply.\n"
    "4. Prefer ordinance names only when strongly implied by the question.\n"
    "5. Do NOT invent specific section numbers, Cap. numbers, deadlines, or "
    "penalties unless provided in the user query.\n"
    "6. If the question is ambiguous, cover 2-3 closely related legal concepts.\n"
    "7. Do NOT answer the user directly. Do NOT give advice.\n"
    "8. Output ONLY the hypothetical legal passage in English."
)

HYDE_NUM_GENERATIONS = int(os.getenv("HYDE_NUM_GENERATIONS", "3"))


async def generate_hyde_passages(
    user_query: str,
    llm,
    num_generations: int = HYDE_NUM_GENERATIONS,
) -> List[str]:
    """
    Generate multiple hypothetical legal passages for a user query using HyDE.
    Returns a list of hypothetical passages (strings).
    """
    messages = [
        SystemMessage(content=HYDE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {user_query}"),
    ]

    async def _generate_one() -> str:
        try:
            response = await llm.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"[HyDE] Error generating passage: {e}")
            return ""

    tasks = [_generate_one() for _ in range(num_generations)]
    passages = await asyncio.gather(*tasks)

    valid_passages = [p for p in passages if p]
    if not valid_passages:
        print("[HyDE] All generations failed, falling back to raw query")
        return [user_query]

    return valid_passages


async def generate_hyde_embeddings(
    user_query: str,
    llm,
    embedding_service,
    num_generations: int = HYDE_NUM_GENERATIONS,
) -> List[float]:
    """
    Full HyDE pipeline: generate hypothetical passages, embed each with the
    document-side prefix, average the embeddings, and return a single vector.

    Also embeds the raw user query and averages it in (dual-path retrieval).

    Returns a single averaged embedding vector ready for Pinecone query.
    """
    passages = await generate_hyde_passages(user_query, llm, num_generations)

    doc_prefix = "Represent this legal document passage for retrieval: "
    hyde_texts = [doc_prefix + p for p in passages]

    query_prefix = "Represent this question for retrieving relevant legal documents: "
    query_text = query_prefix + user_query

    all_texts = hyde_texts + [query_text]
    all_embeddings = embedding_service.embed_documents(all_texts)

    arr = np.array(all_embeddings, dtype=np.float32)
    averaged = np.mean(arr, axis=0)

    norm = np.linalg.norm(averaged)
    if norm > 1e-8:
        averaged = averaged / norm

    return averaged.tolist()


# ---------------------------------------------------------------------------
# Multi-query HyDE: ordinance-decomposition strategy
# ---------------------------------------------------------------------------

MULTI_HYDE_SYSTEM_PROMPT = (
    "You are an expert Hong Kong legal retrieval assistant.\n"
    "Given a user question, generate exactly 5 short hypothetical legal passages "
    "(2-3 sentences each) that resemble excerpts from different Hong Kong ordinances "
    "or regulations relevant to the question.\n\n"
    "Rules:\n"
    "1. Each passage should target a DIFFERENT ordinance or legal provision that could "
    "help answer the question.\n"
    "2. Use formal legal/statutory vocabulary.\n"
    "3. Include likely legal tests, rights, duties, conditions, or exceptions.\n"
    "4. Prefer ordinance names only when strongly implied by the question.\n"
    "5. Do NOT invent specific Cap. numbers, section numbers, deadlines, or penalties "
    "unless provided in the user query.\n"
    "6. Do NOT include analysis, explanations, or commentary — output ONLY the 5 passages.\n"
    "7. Separate each passage with the exact delimiter: ===\n"
    "8. Output exactly 5 passages. No more, no less.\n\n"
    "Format:\n"
    "<passage 1>\n"
    "===\n"
    "<passage 2>\n"
    "===\n"
    "<passage 3>\n"
    "===\n"
    "<passage 4>\n"
    "===\n"
    "<passage 5>"
)

MULTI_HYDE_DELIMITER_RE = re.compile(r"\n\s*===\s*\n")
MULTI_HYDE_MIN_PASSAGE_LEN = 20
MULTI_HYDE_MAX_PASSAGES = 5


async def generate_multi_hyde_passages(
    user_query: str,
    llm,
    max_completion_tokens: int = 1536,
) -> List[str]:
    """
    Ask the LLM to generate up to 5 ordinance-specific hypothetical passages,
    each targeting a different legal provision relevant to the user's question.

    Returns a list of parsed passage strings (1-5 items).
    Returns an empty list on total failure (caller should fall back to old HyDE).
    """
    messages = [
        SystemMessage(content=MULTI_HYDE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {user_query}"),
    ]

    try:
        bound_llm = llm.bind(max_completion_tokens=max_completion_tokens)
        response = await bound_llm.ainvoke(messages)
        raw_text = response.content.strip()
    except Exception as e:
        print(f"[MultiHyDE] LLM generation failed: {e}")
        return []

    if not raw_text:
        print("[MultiHyDE] Empty LLM response")
        return []

    normalized = raw_text.replace("\r\n", "\n")
    blocks = MULTI_HYDE_DELIMITER_RE.split(normalized)

    passages: List[str] = []
    for block in blocks:
        cleaned = block.strip()
        if len(cleaned) >= MULTI_HYDE_MIN_PASSAGE_LEN:
            passages.append(cleaned)

    seen: set[str] = set()
    unique_passages: List[str] = []
    for p in passages:
        if p not in seen:
            seen.add(p)
            unique_passages.append(p)

    unique_passages = unique_passages[:MULTI_HYDE_MAX_PASSAGES]

    if not unique_passages:
        print("[MultiHyDE] No valid passages parsed from LLM response")
        return []

    print(f"[MultiHyDE] Parsed {len(unique_passages)} passages from LLM response")
    return unique_passages
