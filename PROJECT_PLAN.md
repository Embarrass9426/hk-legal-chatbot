# 🧱 Hong Kong Legal RAG System — Architecture & Implementation Plan

## 🎯 Objective
Build a Retrieval-Augmented Generation (RAG) system for Hong Kong legal materials, featuring clickable references, proper legal citations, and traceable source metadata.

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

- **LLM**: DeepSeek 3.2 (via API)
- **Vector Database**: Pinecone
- **Embedding Model**: HuggingFaceEmbedding
- **Backend**: Python (FastAPI + LangChain)
- **Frontend**: React
- **Web Scraping**: PDF Downloading + PDF Parsing (PyMuPDF/pdfplumber)
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
- [ ] Ensure responsive design for legal research on different devices.

### Phase 2: Backend & RAG Pipeline (Python/FastAPI)
- [x] Set up **FastAPI** server.
- [x] Implement **Keyword Extraction** logic to identify target laws/cases from queries.
- [x] Implement **LangChain** retrieval logic with metadata filtering.
- [x] Integrate **DeepSeek 3.2 API** for "Citation-First" generation.
- [ ] Post-process LLM output to map citations to `source_url`.

### Phase 3: Data Ingestion & PDF Parsing (e-Legislation)
- [ ] Download **Employees' Compensation Ordinance (Cap. 282)** PDF from e-Legislation.
- [ ] Implement **PDF Parser** to extract text while preserving section/clause hierarchy.
- [ ] Chunk documents by **Section/Clause** to ensure precise referencing.
- [ ] Extract metadata (Cap No., Section, Clause, Page, URL).

### Phase 4: Vector Database & RAG Refinement
- [ ] Generate embeddings for parsed PDF chunks.
- [ ] Upsert to **Pinecone** with granular metadata.
- [ ] Refine LLM prompt to prioritize **Employee Compensation** scenarios.

---

## 🛠️ Vibe Coding Guidelines
- **Consistency**: Always check `PROJECT_PLAN.md` before adding new data sources.
- **Metadata First**: Never ingest a chunk without a valid `source_url`.
- **Citation Check**: Verify LLM citations against the retrieved metadata before displaying.
