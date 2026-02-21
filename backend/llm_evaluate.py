import os
import json
import asyncio
import numpy as np
from tqdm.asyncio import tqdm
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import VectorStoreManager
from utils import rewrite_query

load_dotenv()

# ============================================================
# CORRECT METRICS
# ============================================================

def precision_at_k(y_true, y_pred, k):
    pred_topk = y_pred[:k]
    rel = sum(1 for doc_id in pred_topk if doc_id in y_true)
    return rel / k


def recall_at_k(y_true, y_pred, k):
    pred_topk = y_pred[:k]
    rel = sum(1 for doc_id in pred_topk if doc_id in y_true)
    return rel / len(y_true) if y_true else 0


def reciprocal_rank(y_true, y_pred):
    for rank, doc_id in enumerate(y_pred, start=1):
        if doc_id in y_true:
            return 1.0 / rank
    return 0.0


# FIXED nDCG (correct ranking-based calculation)
def ndcg_at_k(y_true, retrieved_ids, k):
    dcg = 0.0

    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 3 if doc_id in y_true else 0
        dcg += (2**rel - 1) / np.log2(i + 2)

    ideal_rels = [3] * min(len(y_true), k)
    idcg = sum(
        (2**rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(ideal_rels)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# LLM AUDIT
# ============================================================

async def evaluate_relevance_with_llm(query, context_list, llm):
    context_parts = []
    for i, res in enumerate(context_list):
        meta = res.metadata
        cap_num = meta.get("doc_id", "Unknown Cap")
        section_title = meta.get("section_title", "Unknown Section")
        content = res.page_content if hasattr(res, 'page_content') else ""

        context_parts.append(
            f"--- Search Result {i+1} ---\n"
            f"Cap Number: {cap_num}\n"
            f"Title: {section_title}\n"
            f"Content: {content}"
        )

    context_text = "\n\n".join(context_parts)

    system_prompt = """You are a legal auditor evaluating a RAG system for Hong Kong law.
Score whether the retrieved chunks contain enough information to fully answer the user's legal question.

Scoring:
10 = perfect
7-9 = very good
4-6 = partial
1-3 = insufficient
0 = no content

Respond ONLY in JSON:
{
  "relevance_score": <1-10>,
  "further_info_needed": <true/false>,
  "reasoning": "<brief explanation>"
}
"""

    prompt = f"User Question: {query}\n\nRetrieved Context:\n{context_text}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    try:
        response = await llm.ainvoke(messages)
        res_text = response.content.strip()

        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0].strip()
        elif "```" in res_text:
            res_text = res_text.split("```")[1].strip()

        return json.loads(res_text)

    except Exception as e:
        print(f"LLM audit error: {e}")
        return {
            "relevance_score": 0,
            "further_info_needed": True,
            "reasoning": str(e)
        }


# ============================================================
# MAIN EVALUATION LOGIC
# ============================================================

async def evaluate_pinecone_with_audit(queries, relevance_dict, vs_manager, llm, top_k=10):

    metrics_orig = defaultdict(list)
    metrics_rewritten = defaultdict(list)
    metrics_expanded = defaultdict(list)
    audit_results = []

    print(f"Evaluating {len(queries)} queries...")

    for q in tqdm(queries, desc="Evaluating"):

        q_text = q["query"]
        q_id = str(q["id"])
        relevant_docs = relevance_dict.get(q_id, [])

        # ====================================================
        # ORIGINAL SEARCH
        # ====================================================
        results_orig = vs_manager.search(q_text, k=top_k)

        # FIXED ID FORMAT (matches ingestion)
        retrieved_orig = [
            f"{r.metadata.get('doc_id')}_chunk_{int(r.metadata.get('chunk_index', 0))}"
            for r in results_orig
        ]

        if relevant_docs:
            metrics_orig["MRR"].append(
                reciprocal_rank(relevant_docs, retrieved_orig)
            )
            metrics_orig["Recall@10"].append(
                recall_at_k(relevant_docs, retrieved_orig, 10)
            )
            metrics_orig["nDCG@10"].append(
                ndcg_at_k(relevant_docs, retrieved_orig, 10)
            )

        # ====================================================
        # REWRITTEN SEARCH
        # ====================================================
        rewritten_q = await rewrite_query(q_text, llm)
        results_rew = vs_manager.search(rewritten_q, k=top_k)

        retrieved_rew = [
            f"{r.metadata.get('doc_id')}_chunk_{int(r.metadata.get('chunk_index', 0))}"
            for r in results_rew
        ]

        if relevant_docs:
            metrics_rewritten["MRR"].append(
                reciprocal_rank(relevant_docs, retrieved_rew)
            )
            metrics_rewritten["Recall@10"].append(
                recall_at_k(relevant_docs, retrieved_rew, 10)
            )
            metrics_rewritten["nDCG@10"].append(
                ndcg_at_k(relevant_docs, retrieved_rew, 10)
            )

        # ====================================================
        # EXPANDED SEARCH
        # ====================================================
        results_exp = vs_manager.search_with_expansion(rewritten_q, k=top_k)

        retrieved_exp = []
        for section in results_exp:
            for chunk in section['chunks']:
                m = chunk['metadata']
                chunk_id = f"{m.get('doc_id')}_chunk_{int(m.get('chunk_index', 0))}"
                if chunk_id not in retrieved_exp:
                    retrieved_exp.append(chunk_id)

        if relevant_docs:
            metrics_expanded["MRR"].append(
                reciprocal_rank(relevant_docs, retrieved_exp)
            )
            metrics_expanded["Recall@AllExpanded"].append(
                recall_at_k(relevant_docs, retrieved_exp, len(retrieved_exp))
            )
            metrics_expanded["nDCG@AllExpanded"].append(
                ndcg_at_k(relevant_docs, retrieved_exp, len(retrieved_exp))
            )

        # ====================================================
        # LLM AUDIT
        # ====================================================
        audit_res = await evaluate_relevance_with_llm(q_text, results_rew, llm)

        audit_results.append({
            "query_id": q_id,
            "query": q_text,
            "rewritten_query": rewritten_q,
            "audit": audit_res
        })

    # Safe averaging
    scores = [a["audit"].get("relevance_score", 0) for a in audit_results]
    further = [1 if a["audit"].get("further_info_needed") else 0 for a in audit_results]

    results = {
        "retrieval": {
            "original": {m: np.mean(v) if v else 0 for m, v in metrics_orig.items()},
            "rewritten": {m: np.mean(v) if v else 0 for m, v in metrics_rewritten.items()},
            "expanded": {m: np.mean(v) if v else 0 for m, v in metrics_expanded.items()}
        },
        "audit_summary": {
            "avg_relevance_score": np.mean(scores) if scores else 0,
            "further_info_needed_rate": np.mean(further) if further else 0
        }
    }

    return results, audit_results


# ============================================================
# MAIN
# ============================================================

async def main():
    from langchain_openai import ChatOpenAI

    QUERIES_PATH = "data/queries.jsonl"
    RELEVANCE_PATH = "data/relevance.json"

    queries = [json.loads(line) for line in open(QUERIES_PATH, 'r', encoding='utf-8')]
    relevance_dict = json.load(open(RELEVANCE_PATH, 'r', encoding='utf-8'))

    vs_manager = VectorStoreManager()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model='deepseek-chat',
        api_key=deepseek_api_key,
        base_url='https://api.deepseek.com',
        temperature=0
    )

    results, audit_details = await evaluate_pinecone_with_audit(
        queries, relevance_dict, vs_manager, llm
    )

    print("\n--- Evaluation Results ---")
    print(json.dumps(results, indent=2))

    with open("data/llm_audit_details.json", "w", encoding="utf-8") as f:
        json.dump(audit_details, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())