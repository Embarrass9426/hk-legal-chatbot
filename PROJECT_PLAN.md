# 🧱 Hong Kong Legal RAG System — Architecture & Implementation Plan

## 🎯 Objective
Build a Retrieval-Augmented Generation (RAG) system for all Hong Kong Ordinances (Caps) from e-Legislation, featuring clickable references, proper legal citations, and traceable source metadata.

---

## 📐 System Overview

### 🔁 Query Flow
1. **User Query**: User asks a legal question.
2. **Retrieval**: System searches vector database for relevant HK legal documents (Ordinances, Case Law).
3. **Augmentation**: Context is injected into the LLM prompt along with metadata.
4. **Generation**: LLM generates an answer with inline citations.
5. **Output**: Answer text + Structured, clickable reference links.
6. **Traceability**: Clicking a link takes the user to the original source (HKLII, e-Legislation, etc.).

---

## 🧩 Tech Stack

- **LLM**: DeepSeek-V3.2 (via API)
- **Vector Database**: Pinecone
- **Embedding Model**: `Yuan-embedding-2.0-en` (Hugging Face)
- **Reranking Model**: `Qwen3-Reranker-8B` (Hugging Face)
- **Backend**: Python (FastAPI + LangChain)
- **Frontend**: React (Vite + Tailwind CSS)
- **Web Scraping**: Playwright (for SPAs) + Unstructured (for PDFs)
- **Data Sources**: 
    - [e-Legislation](https://www.elegislation.gov.hk/) (Primary: PDF Statutes)
    - [HKLII](https://www.hklii.hk/) (Secondary: Case Law)

---

## 🗂️ Metadata Management & Schema

To ensure consistency and traceability, every document chunk must follow this schema:

| Field | Description | Example |
| :--- | :--- | :--- |
| `id` | Unique identifier | `hk-ord-cap7-s5` |
| `title` | Full title of the document | `Landlord and Tenant (Consolidation) Ordinance` |
| `source_url` | Direct link to the source | `https://www.elegislation.gov.hk/hk/cap7` |
| `jurisdiction` | Always "Hong Kong" | `Hong Kong` |
| `type` | Ordinance, Case, or Regulation | `Ordinance` |
| `citation` | Standard HK style citation | `Cap. 7, Section 5(1)` |
| `page_number` | If applicable (PDFs) | `12` |
| `date` | Enactment or Judgment date | `2023-01-01` |

---

## 🔗 Link Preprocessing
- **Validation**: Scripts to check URL status (200 OK) before ingestion.
- **Archiving**: Use Wayback Machine or local PDF storage if links are prone to breaking.
- **Normalization**: Ensure URLs are clean and direct.

---

## ⚖️ Citation Formatting Rules

The system must strictly follow standard Hong Kong legal citation styles:

### 1. Case Law
- **Format**: `Party A v Party B [Year] HKCFI Number`
- **Example**: `Chan v Lee [2023] HKCFI 1234`
- **Components**: Parties, Neutral Citation (Year, Court, Case Number).

### 2. Ordinances (Statutes)
- **Format**: `Short Title, Section Number`
- **Example**: `Landlord and Tenant Ordinance, Section 5(1)`
- **Alternative**: `Cap. 7, s. 5`

---

## 🚀 Implementation Roadmap

### Phase 1: Frontend UI/UX (React)
- [x] Design and build the **React** interface.
- [x] Create mock-up for answer display and reference cards.
- [x] Implement clickable reference links (opening in new tabs).
- [x] Ensure responsive design for legal research on different devices.
- [x] Implement **Language Toggle** (English/Traditional Chinese).

### Phase 2: Backend & RAG Pipeline (Python/FastAPI)
- [x] Set up **FastAPI** server.
- [x] Implement **Keyword Extraction** logic to identify target laws/cases from queries.
- [ ] Implement **Smarter Retrieval Pipeline**:
    - [x] **Query Rewriting**: Use LLM to rewrite user query for better legal retrieval context.
    - [x] **Asymmetric Prompting**: Prefix queries and documents with instructional prompts for better mapping.
    - [x] **Reranking**: Integrated `Qwen/Qwen3-Reranker-8B` with 4-bit quantization (Q4) to refine results.
    - [x] **Full Section Context**: For top reranked chunks, fetch all chunks from the same section to provide full legal context to the LLM.
- [x] Implement **LangChain** retrieval logic with metadata filtering.
- [x] Integrate **DeepSeek-V3 API** for "Citation-First" generation.
- [x] Post-process LLM output to map citations to `source_url` (via structured reference streaming).
- [x] Support **Multi-language Generation** (Traditional Chinese/English).

### Phase 3: Data Ingestion & PDF Parsing (e-Legislation)
- [x] Download **Employees' Compensation Ordinance (Cap. 282)** PDF (Initial Test Case).
- [ ] **Cap Discovery**: Scrape e-Legislation index to generate a sorted list of all valid Cap numbers (e.g., 1, 207, 207A).
- [x] **Batch Downloader**: Implement a robust downloader with retry logic to fetch all identified PDFs.
- [x] **Unstructured PDF Parser**:
    - [x] **Element Identification**: Use `unstructured` library to partition PDFs into narrative text, titles, and list items.
    - [x] **Section Reconstruction**: Logic to group elements into legal sections based on title detection and hierarchy.
    - [x] **Section-Based Chunking**:
        - Split document by sections (extracted via `unstructured`).
        - Chunk sections into 300 tokens with 10% overlap using `AutoTokenizer`.
    - [x] **Metadata Enrichment**:
        - Store with schema: `doc_id`, `section_id`, `section_title`, `chunk_index`, `total_chunks_in_section`.
        - Ensure every chunk has a direct URL with physical page anchors (`#page=N`).
- [x] **Storage**: Save each parsed Ordinance as an individual `cap{num}.json` file in `backend/data/parsed/`.

### Phase 4: RAG Implementation & Refinement (Current Focus)
- [x] **Model Selection**: Selected `Yuan-embedding-2.0-en` and `Qwen3-Reranker-8B` for high-precision legal retrieval.
- [x] **Asymmetric Retrieval Pipeline**:
    - [x] Implement **Query Rewriting** to focus on legal concepts using LLM.
    - [x] Apply **Asymmetric Prompting**: Prefix queries with `"Represent this question for retrieving relevant legal documents:"` and chunks with `"Represent this legal document passage for retrieval:"`.
- [x] **Section-Aware Chunking**:
    - [x] Split sections into 300-token chunks with 10% overlap using token-based offset mapping.
    - [x] Tag chunks with `doc_id`, `section_id`, and `chunk_index` for full-section reconstruction.
- [x] **Context Expansion logic**:
    - [x] Retrieve Top 10 -> Deduplicate -> Rerank Top 5 using `Qwen3-Reranker-8B` (Q4).
    - [x] For each Top 5 chunk, fetch **all sibling chunks** from the same `section_id` to provide complete legal context to DeepSeek.
- [ ] **Retrieval Evaluation**:
    - [ ] Develop evaluation script using **Recall@K** and **MRR** (Mean Reciprocal Rank) metrics.
    - [ ] Create a golden dataset of query-section pairs for benchmarking.
- [ ] **Multi-turn Conversation & Memory**:
    - [ ] Implement a **Conversation Buffer** with context window management (sliding window or summarization).

### Phase 5: Production Scaling
- [x] **Batch Downloader**: Implement a robust downloader with retry logic to fetch all 3,145 identified PDFs.
- [ ] **Batch Ingestor**: Script to iterate through `backend/data/parsed/*.json` and upsert to Pinecone.
- [ ] **Vector Database Migration**: Scale Pinecone index to handle the full corpus of HK Ordinances.
- [x] **OCR & Performance Optimization**:
    - [x] **GPU Acceleration**: Enabled CUDA/TensorRT for ONNX.
    - [x] **New OCR Stack**: Replaced Tesseract with **PaddleOCR** and **YOLOX** for faster, layout-aware parsing.
    - [x] **Auto-Tuning**: Developed `optimize_ingestion.py` to benchmark concurrency and batching parameters.
- [ ] Implement caching for scraped/parsed content to reduce latency.
- [ ] Expand data ingestion to include Case Law from HKLII.
- [ ] Implement an in-app PDF viewer for seamless reference checking.

---

## 🛠️ Vibe Coding Guidelines
- **Consistency**: Always check `PROJECT_PLAN.md` before adding new data sources.
- **Metadata First**: Never ingest a chunk without a valid `source_url`.
- **Citation Check**: Verify LLM citations against the retrieved metadata before displaying.
- **Context Expansion**: Always provide the full legal section to the LLM if the section ID is identified.
