# backend/evaluate_models.py
import json
import numpy as np
import os
import asyncio
from tqdm.asyncio import tqdm
from collections import defaultdict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import the boosted model and rewriting logic
from vector_store import VectorStoreManager
from utils import rewrite_query

load_dotenv()

# --- 1. Data Loading ---------------------------------------------------------

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- 2. Evaluation Metrics ---------------------------------------------------

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

def ndcg_at_k(y_true, y_scores, k):
    """y_scores = dict {doc_id: relevance_score}"""
    if not any(score > 0 for score in y_scores.values()):
        return 0.0
        
    ideal = sorted(y_scores.values(), reverse=True)[:k]
    dcg = 0
    for i, (doc_id, rel_i) in enumerate(list(y_scores.items())[:k]):
        dcg += (2**rel_i - 1) / np.log2(i + 2)
        
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

# --- 3. Pipeline -------------------------------------------------------------

async def evaluate_pipeline(queries, relevance_dict, vs_manager, llm, top_k=10):
    metrics_orig = defaultdict(list)
    metrics_rewritten = defaultdict(list)
    metrics_expanded = defaultdict(list)
    print(f"Evaluating {len(queries)} queries using Pinecone index...")

    for q in tqdm(queries, desc="Evaluating"):
        q_text, q_id = q["query"], str(q["id"])
        relevant_docs = relevance_dict.get(q_id, [])
        if not relevant_docs:
            continue

        # --- 1. Without Rewrite ---
        results_orig = vs_manager.search(q_text, k=top_k)
        retrieved_orig = [
            f"{r.metadata.get('doc_id')}_{int(r.metadata.get('chunk_index', 0))}_{int(r.metadata.get('page_number', 0))}"
            for r in results_orig
        ]
        
        metrics_orig["MRR"].append(reciprocal_rank(relevant_docs, retrieved_orig))
        metrics_orig["Recall@10"].append(recall_at_k(relevant_docs, retrieved_orig, 10))
        rel_scores_orig = {doc_id: (3 if doc_id in relevant_docs else 0) for doc_id in retrieved_orig}
        metrics_orig["nDCG@10"].append(ndcg_at_k(relevant_docs, rel_scores_orig, 10))

        # --- 2. With Rewrite ---
        search_query = await rewrite_query(q_text, llm)
        results_rew = vs_manager.search(search_query, k=top_k)
        retrieved_rewritten = [
            f"{r.metadata.get('doc_id')}_{int(r.metadata.get('chunk_index', 0))}_{int(r.metadata.get('page_number', 0))}"
            for r in results_rew
        ]

        metrics_rewritten["MRR"].append(reciprocal_rank(relevant_docs, retrieved_rewritten))
        metrics_rewritten["Recall@10"].append(recall_at_k(relevant_docs, retrieved_rewritten, 10))
        rel_scores_rewritten = {doc_id: (3 if doc_id in relevant_docs else 0) for doc_id in retrieved_rewritten}
        metrics_rewritten["nDCG@10"].append(ndcg_at_k(relevant_docs, rel_scores_rewritten, 10))

        # --- 3. With Expansion (on Rewritten) ---
        results_exp = vs_manager.search_with_expansion(search_query, k=top_k)
        retrieved_exp = []
        for section in results_exp:
            for chunk in section['chunks']:
                m = chunk['metadata']
                chunk_id = f"{m.get('doc_id')}_{int(m.get('chunk_index', 0))}_{int(m.get('page_number', 0))}"
                if chunk_id not in retrieved_exp:
                    retrieved_exp.append(chunk_id)

        metrics_expanded["MRR"].append(reciprocal_rank(relevant_docs, retrieved_exp))
        metrics_expanded["Recall@AllExpanded"].append(recall_at_k(relevant_docs, retrieved_exp, len(retrieved_exp)))
        rel_scores_exp = {d: 3 if d in relevant_docs else 0 for d in retrieved_exp}
        metrics_expanded["nDCG@AllExpanded"].append(ndcg_at_k(relevant_docs, rel_scores_exp, len(retrieved_exp)))

    results = {
        "original": {m: np.mean(v) if v else 0 for m, v in metrics_orig.items()},
        "rewritten": {m: np.mean(v) if v else 0 for m, v in metrics_rewritten.items()},
        "expanded": {m: np.mean(v) if v else 0 for m, v in metrics_expanded.items()}
    }
    return results

# --- 4. Execution ------------------------------------------------------------

async def main():
    # Paths
    QUERIES_PATH = "data/queries.jsonl"
    RELEVANCE_PATH = "data/relevance.json"

    # Load data
    queries = load_jsonl(QUERIES_PATH)
    
    if os.path.exists(RELEVANCE_PATH):
        with open(RELEVANCE_PATH, "r") as f:
            relevance_dict = json.load(f)
    else:
        relevance_dict = {}

    print(f"Loaded {len(queries)} queries.")

    # Initialize components
    vs_manager = VectorStoreManager()
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        model='deepseek-chat', 
        api_key=api_key, 
        base_url='https://api.deepseek.com',
        temperature=0
    )

    # Run evaluation pipeline (Async)
    results = await evaluate_pipeline(
        queries, 
        relevance_dict, 
        vs_manager,
        llm=llm
    )
    
    print("\n--- Evaluation Results (Pinecone) ---")
    print(json.dumps(results, indent=2))
    
    if not relevance_dict:
        print("\nNote: Populate 'backend/data/relevance.json' with ground truth mapping to get non-zero metrics.")

if __name__ == "__main__":
    asyncio.run(main())
