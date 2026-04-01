import os
import json
from langchain_core.messages import HumanMessage, SystemMessage


async def rewrite_query(user_query: str, llm, scenario: str = "") -> str:
    """
    Rewrites the user question into a legal-focused retrieval query.
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


class LegalReranker:
    """
    Reranker using Qwen3/2-Reranker-8B or similar BGE models.
    Optimizes top-K results to ensure most relevant legal sections are prioritized.
    """

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        if self.model is None:
            try:
                from FlagEmbedding import FlagReranker
                import torch

                # Only load if CUDA is available as 8B rerankers are intensive
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = FlagReranker(self.model_name, use_fp16=(device == "cuda"))
                print(f"Reranker {self.model_name} loaded successfully on {device}.")
            except Exception as e:
                print(f"Failed to load reranker: {e}")
                self.model = False

    def rerank(self, query, contexts, top_n=3):
        """
        Reranks a list of contexts based on relevance to the query.
        """
        self._load_model()
        if not self.model or not contexts:
            return contexts[:top_n]

        pairs = [[query, ctx["page_content"]] for ctx in contexts]
        scores = self.model.compute_score(pairs)

        scored_contexts = list(zip(scores, contexts))
        scored_contexts.sort(key=lambda x: x[0], reverse=True)

        return [ctx for score, ctx in scored_contexts[:top_n]]
