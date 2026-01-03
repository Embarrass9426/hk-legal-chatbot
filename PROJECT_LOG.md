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

---
*Log updated on 2026-01-01*

## 📅 January 2, 2026

### ✅ Completed Tasks
- **LLM-Powered PDF Parsing**:
    - Integrated **DeepSeek LLM** into the PDF parsing pipeline to solve regex-based extraction failures.
    - Implemented a **Two-Step LLM Process** for Table of Contents (TOC) extraction:
        1. **Identification**: LLM scans raw TOC text to generate a clean, plain-text list of all sections and schedules.
        2. **Structuring**: LLM matches the identified list to titles and page labels, outputting a structured JSON.
    - Added **Verification & Recovery Logic**: The script now compares the LLM's JSON output against the initial identification list and triggers a targeted retry for any missing sections.
    - Improved **Content Extraction**:
        - Fixed "0-length content" issues by ensuring at least one page is read when multiple sections share a page.
        - Implemented **Header-Based Splitting**: Uses regex to find the exact start and end of a section on a shared page, discarding text from adjacent sections.
        - Increased TOC scan range to 15 pages to support larger ordinances.
- **Multi-language Support**:
    - Added a **Language Toggle** (EN/繁) to the frontend header.
    - Implemented dynamic initial greeting translation based on selected language.
    - **UI Localization**: Updated the header title, subtitle, input placeholder, and error messages to switch between English and Traditional Chinese based on the toggle.
    - Updated backend `ChatRequest` schema to accept `language` parameter.
    - Refined LLM system prompt to enforce response language (Traditional Chinese or English) while maintaining legal citation integrity.
    - Ensured the RAG pipeline correctly handles language preferences during the generation phase.
- **Testing Infrastructure**:
    - Created `TEST_QUESTIONS.md` containing factual, scenario-based, and edge-case questions specifically for Cap. 282 to evaluate RAG performance.
- **Environment & Dependencies**:
    - Configured and verified the Python virtual environment (`.venv`) for consistent execution.
    - Installed and configured **Playwright Chromium** for automated PDF downloads.

### 🛠️ Current Status
- **Backend**: PDF parsing is now highly accurate, successfully capturing 109+ sections of Cap. 282 with precise content boundaries.
- **Frontend**: Users can now toggle between English and Traditional Chinese.
- **Next Steps**: 
    - Evaluate chatbot performance using the new test questions.
    - Implement multi-ordinance support in the ingestion pipeline.

---
*Log updated on 2026-01-02*

## 📅 January 4, 2026

### ✅ Completed Tasks
- **Strategic Pivot to "All Caps"**:
    - Expanded the project scope from a single Ordinance (Cap. 282) to all **3,000+ Hong Kong Ordinances** available on e-Legislation.
    - Updated `PROJECT_PLAN.md` to reflect the new "all Caps" strategy and the shift towards an OCR-ready, hybrid parsing approach.
- **Cap Discovery & Indexing**:
    - Developed `backend/scripts/cap_discovery.py` using **Playwright** to scrape the e-Legislation Chapter Number Index.
    - Successfully identified and indexed **3,086 unique Cap numbers** (including alphanumeric ones like 1A, 207A).
    - Generated `backend/data/cap_list.json` as the master list for batch processing.
- **Intelligent PDF Parser (v2)**:
    - Re-engineered `backend/pdf_parser.py` into a robust `PDFLegalParser` class.
    - **TOC Detection**: Integrated DeepSeek LLM to analyze the first 40 pages of any PDF to determine if a Table of Contents exists.
    - **Hybrid Extraction Pipeline**:
        - **Branch A (TOC)**: Uses a 2-step LLM process (Identification -> Structuring) to map sections to page labels, followed by regex-refined content extraction.
        - **Branch B (Fallback)**: Implemented a full-text extraction fallback for shorter documents or those without a TOC.
    - **Granular Storage**: Configured the parser to save results as individual `cap{num}.json` files in `backend/data/parsed/` to support incremental vector database updates.
- **Documentation & Architecture**:
    - Created `docs/PDF_PARSER_LOGIC.md` to provide a detailed technical specification and pseudocode for the parsing pipeline.
    - Implemented a custom sorting algorithm for alphanumeric Cap numbers to ensure logical processing order.

### 🛠️ Current Status
- **Data**: We now have a complete list of all HK Ordinances to be ingested.
- **Parser**: The intelligent parsing pipeline is ready for batch execution.
- **Next Steps**: 
    - Implement a **Batch Downloader** with retry logic for the 3,000+ PDFs.
    - Develop a **Batch Ingestor** to upsert the parsed JSON files into Pinecone.
    - Refine OCR fallback for scanned/image-based PDFs.

---
*Log updated on 2026-01-04*

