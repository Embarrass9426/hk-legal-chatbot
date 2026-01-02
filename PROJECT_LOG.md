# 📝 Project Log — HK Legal RAG System

## 📅 December 29, 2025

### ✅ Completed Tasks
- **Project Initialization**: Established the core project structure with `frontend/` and `backend/` directories.
- **Tech Stack Pivot**: Switched the backend from Node.js to **Python (FastAPI + LangChain)** to better support AI agent development and the existing Python scraping logic.
- **Backend Setup**:
    - Initialized a **FastAPI** server in `backend/main.py`.
    - Integrated **DeepSeek-V3** LLM using LangChain's `ChatOpenAI` compatible interface.
    - Implemented **Streaming Responses** (SSE) to provide real-time text generation.
    - Configured environment variable management via `.env`.
- **Frontend Development**:
    - Built a responsive **React** chat interface with Tailwind CSS.
    - Implemented **Dark Mode** support.
    - Integrated **Streaming Reader** to consume real-time updates from the FastAPI backend.
    - Added a dynamic **Loading Indicator** (typing bubble) that hides once the stream begins.
    - Integrated **Markdown Rendering** using `react-markdown` and `remark-gfm` to support bold text, headers, and lists.
    - Added custom CSS typography for legal document readability.
- **Security & Git**:
    - Initialized Git repository.
    - Configured `.gitignore` files in root, `backend/`, and `frontend/` to prevent accidental exposure of `.env` files and API keys.
    - Verified that sensitive files are correctly ignored by Git.

### 🛠️ Current Status
- **Frontend**: UI is fully functional and connected to the backend.
- **Backend**: Basic LLM chat is working with streaming.
- **Next Steps**: 
    - Begin Phase 3: Web Scraping (Beautiful Soup) for HK Ordinances.
    - Begin Phase 4: Vector Database (Pinecone) integration for RAG.

---
*Log created on 2025-12-29*

## 📅 December 31, 2025

### ✅ Completed Tasks
- **HKLII Investigation**: 
    - Created `backend/scripts/inspect_hklii.py` to analyze HKLII's HTML structure.
    - Identified that HKLII is a **Vuetify SPA**, necessitating a shift from static scraping to **Playwright** for dynamic content rendering.
- **Keyword Extraction**:
    - Implemented `backend/utils.py` using DeepSeek to extract `target_law`, `section`, and `keywords` from user queries.
- **Dynamic Scraper**:
    - Built `backend/scraper.py` using Playwright to fetch, parse, and chunk ordinances (e.g., Cap 1) directly from HKLII.
- **Vector Store Integration**:
    - Implemented `backend/vector_store.py` using **Pinecone** for vector storage and **HuggingFace** (`all-MiniLM-L6-v2`) for local embeddings.
    - Configured automatic index creation and document upserting.
- **Full RAG Workflow Integration**:
    - Updated `backend/main.py` to orchestrate the dynamic RAG pipeline: Query Analysis -> Scraping (if needed) -> Vector Search -> Augmented Generation.
    - Implemented reference streaming to the frontend.
- **Frontend Enhancements**:
    - Updated `ChatInterface.jsx` to handle and display structured `references` using the `ReferenceCard` component.
- **Documentation**:
    - Created `GITHUB_WORKFLOW.md` to provide a quick reference for Git commands and project workflow.
- **Pivot to PDF-based RAG**:
    - Shifted primary data source to **e-Legislation** PDFs per senior's suggestion.
    - Focused initial use case on **Employees' Compensation Ordinance (Cap. 282)**.
    - Updated project plan to include PDF parsing and section-based chunking.
    - Implemented `backend/pdf_parser.py` using **Playwright** for downloading and **PyMuPDF** for parsing.
    - Successfully ingested 157 sections of Cap. 282 into Pinecone.
    - Refined LLM system prompt and keyword extraction for Employee Compensation scenarios.
- **UI/UX Refinements**:
    - **Loading Animation Timing**: Fixed the loading indicator logic to remain visible until the server is ready to stream the output, preventing a "frozen" UI during backend processing.
    - **Empty Bubble Fix**: Implemented conditional rendering to prevent empty assistant chat bubbles from appearing while the RAG pipeline is running.
    - **Streaming Stability**: Improved SSE (Server-Sent Events) handling in the frontend to ensure smooth transitions between "loading" and "answering" states.

### 🛠️ Current Status
- **Frontend**: Now displays clickable reference cards for legal citations and has a polished loading experience.
- **Backend**: Fully functional RAG pipeline with dynamic scraping and PDF parsing capabilities.
- **Next Steps**: 
    - Refine scraper selectors for more granular section extraction.
    - Implement caching to avoid redundant scraping of the same ordinances.
    - Expand scraping to include Case Law databases on HKLII.

---
*Log updated on 2025-12-31*

## 📅 January 1, 2026

### ✅ Completed Tasks
- **Enhanced PDF Parsing**:
    - Implemented a robust **TOC-to-Page mapping** logic in `backend/pdf_parser.py`.
    - Hardcoded TOC extraction from pages 1-9 to build a reliable section index.
    - Developed a manual page label scanner to map printed labels (e.g., "3A-10") to physical PDF indices, overcoming empty metadata in source PDFs.
    - Improved title extraction to handle multi-line headers and filter out irrelevant text.
    - Added logic to filter out "repealed" or "omitted" sections from the database.
    - Added detailed console logging to display the list of parsed sections during ingestion.
    - Implemented automatic content chunking in `backend/vector_store.py` using `RecursiveCharacterTextSplitter` to stay within Pinecone's 40KB metadata limit.
- **Vector Store Refresh**:
    - Re-ingested Cap. 282 (Employees' Compensation Ordinance) with 73 high-quality sections, each mapped to its exact printed page label and physical page.
- **Frontend Reference Improvements**:
    - Updated `ReferenceCard.jsx` to be fully clickable.
    - Added "Page X" indicators to reference cards for better transparency.
    - Improved hover states and visual feedback for legal citations.

### 🛠️ Current Status
- **Frontend**: Reference cards now link directly to the exact page of the Ordinance PDF.
- **Backend**: Metadata in Pinecone now includes granular page-level information.
- **Next Steps**: 
    - Implement a PDF viewer directly in the app (optional).
    - Expand the database to include more Ordinances related to labor law.

---
*Log updated on 2026-01-01*
