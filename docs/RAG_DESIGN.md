# 🧠 RAG Implementation Design & Pseudocode

This document outlines the specialized RAG pipeline using `Yuan-embedding-2.0-en` and `Qwen3-Reranker-8B`.

---

## 🛠️ Data Ingestion Functions

### 1. `split_into_sections(pdf_path)`
**Purpose**: Uses `pdf_parser.py` to identify legal sections (e.g., Section 1, Section 2) from the Ordinance PDF.
**Returns**: List of section objects with text and section-level metadata.

### 2. `chunk_section(section_text, section_metadata)`
**Purpose**: Chunks a single section into smaller pieces for vector search.
**Strategy**: 300 tokens per chunk with 10% overlap (30 tokens).
**Metadata Schema**:
```json
{
  "doc_id": "string",
  "section_id": "string",
  "section_title": "string",
  "chunk_index": int,
  "total_chunks_in_section": int,
  "citation": "string (e.g. Cap. 282, s. 12)"
}
```

### 3. `get_doc_embedding(chunk_text)`
**Purpose**: Generates asymmetric embedding for storage.
**Model**: `Yuan-embedding-2.0-en`
**Format**:
```python
doc_text = f"Represent this legal document passage for retrieval:\n{chunk_text}"
embedding = embed_model.encode(doc_text)
```

---

## 🔍 Retrieval & Generation Functions

### 4. `rewrite_query(user_query)`
**Purpose**: Refines the user question into a format optimized for legal document retrieval.
**Prompt**:
```text
Rewrite the following user question to improve retrieval of relevant legal documents.
Preserve the original intent.
Do not add assumptions, jurisdictions, or conclusions.
Focus on key legal concepts and terminology.

User question:
{user_query}
```

### 5. `get_query_embedding(rewritten_query)`
**Purpose**: Generates asymmetric embedding for inquiry.
**Model**: `Yuan-embedding-2.0-en`
**Format**:
```python
query_text = f"Represent this question for retrieving relevant legal documents:\n{user_query}"
query_embedding = embed_model.encode(query_text)
```

### 6. `retrieve_top_k(query_embedding, k=10)`
**Purpose**: Fetches the initial candidate set from Pinecone.
**Steps**:
1. Search Pinecone for `k=10`.
2. Filter out exact duplicate `section_id` results if necessary.

### 7. `rerank_results(query, retrieved_chunks, n=5)`
**Purpose**: Re-orders the top K results using a cross-encoder for higher precision.
**Model**: `Qwen3-Reranker-8B`
**Input**: Query + List of Chunks.
**Output**: Top 5 most relevant chunks.

### 8. `expand_to_full_sections(top_chunks)`
**Purpose**: For the top 5 chunks, retrieve **all** sibling chunks belonging to the same `section_id` to provide full legal context.
**Algorithm**:
```python
all_context = []
for chunk in top_chunks:
    if chunk.section_id not in processed_sections:
        # Fetch all chunks where section_id == chunk.section_id from DB
        full_section_text = db.fetch_all_chunks(chunk.section_id)
        all_context.append(full_section_text)
        processed_sections.add(chunk.section_id)
```

---

## 📄 Main RAG Pipeline Pseudocode

```python
def rag_pipeline(user_query):
    # 1. Query Preprocessing
    rewritten_query = rewrite_query(user_query)
    
    # 2. Embedding (Asymmetric)
    query_vector = get_query_embedding(rewritten_query)
    
    # 3. Initial Search
    candidates = vector_db.search(query_vector, top_k=10)
    
    # 4. Deduplication
    unique_candidates = remove_duplicates(candidates)
    
    # 5. Reranking
    reranked_docs = reranker.rank(
        query=rewritten_query, 
        docs=unique_candidates, 
        top_n=5
    )
    
    # 6. Context Expansion
    # We want the full section content, not just the 300-token chunk
    final_context = []
    for doc in reranked_docs:
        section_content = db.get_full_text_by_section_id(doc.metadata['section_id'])
        final_context.append(section_content)
    
    # 7. Generation
    # Inject full sections into the prompt
    answer = llm.generate_answer(
        context=final_context, 
        query=user_query
    )
    
    return answer
```

---

## 📈 Retrieval Evaluation Functions

### 9. `evaluate_retrieval(test_cases, k=10)`
**Purpose**: Benchmarks the retrieval pipeline using objective metrics.
**Metrics**:
- **Recall@K**: Percentage of queries where the correct section appears in the top K results.
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal of the rank of the first relevant document.
    - $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$

**Pseudocode**:
```python
def evaluate_retrieval(test_dataset):
    # test_dataset: list of { "query": str, "expected_section_id": str }
    recall_at_k = []
    reciprocal_ranks = []
    
    for item in test_dataset:
        query = item['query']
        target_id = item['expected_section_id']
        
        # 1. Run full retrieval pipeline
        candidates = retrieve_top_k(get_query_embedding(rewrite_query(query)), k=10)
        
        # 2. Calculate Recall@K
        # Check if the target section exists in the candidates
        found = any(c.metadata['section_id'] == target_id for c in candidates)
        recall_at_k.append(1 if found else 0)
        
        # 3. Calculate MRR
        rank = 0
        for i, c in enumerate(candidates):
            if c.metadata['section_id'] == target_id:
                rank = i + 1
                break
        
        reciprocal_ranks.append(1/rank if rank > 0 else 0)
        
    return {
        "Mean Recall@10": sum(recall_at_k) / len(test_dataset),
        "MRR": sum(reciprocal_ranks) / len(test_dataset)
    }
```

