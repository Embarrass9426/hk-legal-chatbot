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
- **Data**: We now have a complete list of all 3,145 HK Ordinances to be ingested.
- **Parser**: The intelligent parsing pipeline is ready, but implementation is paused for RAG optimization research.
- **Next Steps**: 
    - Research fixed-length chunking with metadata-linked retrieval.
    - Evaluate embedding models for legal text.
    - Implement HyDE and Synthetic QA generation for better retrieval.
    - Develop a memory management system for multi-turn conversations.

---
*Log updated on 2026-01-04*

## 📅 January 4, 2026 (Evening Update)

### 🔍 Major Issues Identified
- **Chunking Inefficiency**: TOC-based parsing creates variable-length sections. Since embedding vectors have fixed lengths, a single section might be split across multiple vectors, leading to fragmented retrieval and poor search results.
- **Embedding Model Suitability**: Current model (`all-MiniLM-L6-v2`) may not be optimal for the nuances of legal terminology and formal sentence structures.
- **Semantic Gap**: User queries often use natural language that differs significantly from the formal, structured language of legal documents, causing low similarity scores in vector search.
- **Lack of Conversation Memory**: The current system is stateless and cannot handle multi-turn dialogues or remember context from previous questions.

### 🚀 Strategic Pivot & Research Decisions
- **Fixed-Length Chunking**: Decided to move towards fixed-length chunks with metadata (section labels, order index). This allows for consistent embedding while enabling the retrieval of the full section by indexing back to related chunks.
- **Query Alignment Research**:
    - **Synthetic QA**: Plan to generate question-answer pairs for each chunk to search against user queries.
    - **HyDE (Hypothetical Document Embeddings)**: Plan to use LLM-generated "fake answers" to bridge the semantic gap between queries and legal text.
- **Memory Management**: Researching sliding window and summarization techniques to maintain conversation context within token limits.
- **Implementation Pause**: Paused the batch downloader and ingestor to ensure the RAG architecture is optimized before processing 3,000+ documents.

---
*Log updated on 2026-01-04*

## 📅 January 7, 2026

###  Completed Tasks
- **Model Research**: Identified `Yuan-embedding-2.0-en` for embeddings and `Qwen3-Reranker-8B` for reranking.
- **RAG Pipeline Redesign**:
    - Defined a new chunking strategy: Section splitting -> 300 token chunks -> 10% overlap.
    - Updated metadata schema to include `doc_id`, `section_id`, `section_title`, `chunk_index`, and `total_chunks_in_section`.
    - Implemented **Asymmetric Prompting** strategy for both queries and documents.
    - Added **Query Rewriting** step to improve retrieval accuracy.
    - Optimized retrieval flow: Top 10 retrieval -> Duplicate removal -> Top 5 reranking -> Full section expansion.
    - **Plan Alignment**: Updated Phase 4 of [PROJECT_PLAN.md](PROJECT_PLAN.md) to align with the specific choice of Yuan-embedding and Qwen3-Reranker, replacing general research goals with concrete implementation steps (Section reconstruction, asymmetric prompting, etc.).

###  Current Status
- **Documentation**: Updated [PROJECT_PLAN.md](PROJECT_PLAN.md) and [PROJECT_LOG.md](PROJECT_LOG.md) with the new architecture.
- **Design**: Created [docs/RAG_DESIGN.md](docs/RAG_DESIGN.md) with detailed function pseudocode.

###  Next Steps
- Implement the new RAG pipeline in [backend/vector_store.py](backend/vector_store.py) and [backend/main.py](backend/main.py).
- Update [backend/pdf_parser.py](backend/pdf_parser.py) to support the new section-based chunking logic.

- **Retrieval Metrics**: Added a new evaluation stage to the project plan and RAG design. We will use **Recall@K** and **MRR** to benchmark our retrieval performance against a golden dataset.

---
*Log updated on 2026-01-07*

## 📅 January 8, 2026

### ✅ Completed Tasks
- **Batch PDF Downloader**:
    - Implemented `backend/scripts/batch_download.py` to automatically download all available English PDFs of HK ordinances from e-Legislation.
    - Handles pagination, skips repealed/cancelled ordinances, and reports any failed downloads at the end.
    - Supports configurable concurrency for efficient and polite scraping.

### 🛠️ Current Status
- **Data Ingestion**: 3,145 HK Ordinances PDF download tasks queued.
- **Next Steps**: 
    - Monitor and verify the completion of the PDF downloading process.
    - Proceed with the ingestion of downloaded PDFs using the `unstructured` library.

---
*Log updated on 2026-01-08*

## 📅 January 8, 2026 (Evening)

### ✅ Completed Tasks
- **Pivot to Unstructured Library**:
    - Decided to move away from LLM-based PDF parsing for section mapping.
    - Adopted the **Unstructured** Python library as the primary engine for PDF content extraction.
    - This change aims for higher accuracy in identifying structural elements (Titles, Tables, List Items) without the latency and cost of LLM calls.
- **Updated Project Plan**:
    - Reflected the shift to `unstructured` in Phase 3 of the [PROJECT_PLAN.md](PROJECT_PLAN.md).
    - Removed previous LLM-driven TOC mapping logic from the planned roadmap.

### 🛠️ Current Status
- **Parsing Strategy**: Unified around `unstructured` for all PDF ordinances.
- **Next Steps**: 
    - Install `unstructured` and its dependencies (libmagic, poppler, etc.).
    - Update `backend/pdf_parser.py` to use `unstructured.partition.pdf`.
    - Implement element-to-section grouping logic.

*Log updated on 2026-01-08*

## 📅 January 9, 2026

### ✅ Completed Tasks
- **OCR & Dependency Fixes**:
    - Resolved "Tesseract not found" by explicitly adding `C:\Program Files\Tesseract-OCR` to the runtime environment `PATH` in `pdf_parser_v2.py`.
    - Integrated logic to handle local **Poppler** binary paths within the project structure for Windows portability.
- **Robust Path Resolution**:
    - Updated `ingest_legal_pdfs.py` to automatically detect PDF directories across both local and root execution contexts.
    - Added support for `--cap` command-line argument for targeted processing of specific ordinances.
- **Database Stability**:
    - Fixed a critical Pinecone error (`Invalid value for id`) by implementing ID truncation for exceptionally long section titles in `vector_store.py`.
    - Suppressed `max_size` deprecation warnings from nested library calls to maintain clean logs during batch ingestion in `pdf_parser_v2.py`.
- **Optimization**:
    - Explicitly disabled the 8B Reranker while preserving the section-expansion retrieval logic to improve performance and reduce VRAM bottlenecks.
- **Automated Data Maintenance**:
    - Implemented `delete_by_doc_id` in `VectorStoreManager` to clear old segments before re-indexing.
    - Updated ingestion pipeline to automatically purge existing vectors for a specific ordinance before uploading new ones, preventing "ghost chunks" when chunking logic changes.
- **GPU Acceleration Fix**:
    - Resolved "not using GPU" issue by identifying and removing the CPU-only `onnxruntime` conflicts.
    - Forced re-installation of `onnxruntime-gpu`.
    - Implemented `os.add_dll_directory` logic in `pdf_parser_v2.py` to point explicitly to `torch\lib` and `nvidia` CUDA DLLs, ensuring `CUDAExecutionProvider` is detected by ONNX Runtime on Windows.

### 🛠️ Current Status
- **Ingestion**: Working reliably for ordinances with long titles (e.g., Cap A1).
- **Environment**: Backend fully configured with Tesseract, Poppler, and the necessary Python venv.

### 📅 Next Steps
- [ ] Run full ingestion for all Employment and Workers' Compensation related ordinances.
- [ ] Verify retrieval quality using the `search_with_expansion` method.

---
*Log updated on 2026-01-09*

## 📅 January 10, 2026

### ✅ Completed Tasks
- **Infrastructure & GPU Acceleration**:
    - **GPU Provider Fixed**: Successfully enabled `CUDAExecutionProvider` and `TensorrtExecutionProvider` in `onnxruntime` by resolving library conflicts and configuring Windows DLL paths.
    - **OCR Stack Upgrade**: Replaced Tesseract with **PaddleOCR (GPU)** and **PaddlePaddle-GPU** (nightly cu129) for significantly faster and higher-fidelity text extraction.
    - **Layout Analysis**: Switched to **YOLOX** as the primary layout model for legal document architectural detection.
- **Parser & Ingestion Optimization**:
    - **Advanced Parallelization**: Updated `pdf_parser_v2.py` to use `multiprocess=True` with 8 physical cores.
    - **Smart Skip Logic**: Implemented check for existing JSON files in `ingest_legal_pdfs.py` to skip processing and embedding for already parsed documents, with a `--force` override for re-indexing.
    - **GPU Diagnostic Logging**: Integrated PaddlePaddle GPU availability checks directly into `pdf_parser_v2.py` to ensure hardware acceleration is active before heavy OCR tasks.
    - **Dynamic Batching**: Implemented `layout_batch_size` parameter in `partition_pdf` to maximize VRAM utilization on the RTX 4060 Ti.
    - **Benchmarking Suite**: Developed `backend/optimize_ingestion.py` to automate performance testing.
    - **Triple-Parameter Sweep**: Instrumented the ingestion pipeline to benchmark **Concurrency** (3-10), **Embedding Batch** (16, 32, 64), and **Layout Batch** (4, 8, 16) combinations.
- **Dependency Consolidation**:
    - Updated `backend/requirements.txt` with production-grade GPU libraries: `onnxruntime-gpu`, `tensorrt`, `paddlepaddle-gpu`, `paddleocr`, and `yolox`.

### 🛠️ Current Status
- **Environment**: High-performance GPU ingestion environment is fully stabilized.
- **Performance**: The system is now optimized for the high-throughput ingestion of the 3,000+ ordinance corpus.

### 📅 Next Steps
- [ ] Record and implement the "Best Parameters" found by the optimization script.
- [ ] Kick off the full batch ingestion for the remaining 3,000+ Ordinances.
- [ ] Implement **Recall@K** evaluation script to benchmark retrieval accuracy after the full corpus is ingested.

---
*Log updated on 2026-01-10*

## 📅 January 17, 2026

### ✅ Completed Tasks
- **Semantic Chunking V2**:
    - Implemented `chunk_section_paragraph_based` in `pdf_parser_v2.py`, replacing fixed-size chunking with paragraph-merging logic based on semantic similarity (Threshold: 0.8) up to a 1200-token limit.
    - Switched to high-fidelity CLS pooling for `IEITYuan/Yuan-embedding-2.0-en` to improve embedding representation.
- **Boosted GPU Inference Pipeline**:
    - **ONNX/TensorRT Migration**: Integrated `optimum` and `onnxruntime-gpu` with `TensorrtExecutionProvider` to achieve line-rate embedding speeds.
    - **Architecture Support (RTX 50-series)**: Resolved critical "sm_120" compatibility issues and GPU kernel errors by updating the environment to **PyTorch 2.9.1+cu128** and CUDA 12.8.
    - **ONNX Input Engineering**: Implemented manual generation of `position_ids` within the inference loop to satisfy the requirements of the exported Yuan model graph.
- **Environment & Reliability**:
    - **uv Migration**: Fully adopted `uv` for streamlined dependency management and environment stability.
    - **Windows DLL Injection**: Hardened the programmatic `PATH` and `add_dll_directory` logic across all backend scripts to ensure `nvinfer_10.dll` and other CUDA libraries are correctly loaded.
    - **Pinecone Zero-Vector Fix**: Implemented protection against empty/non-semantic text segments by adding epsilon noise to zero-vectors, preventing `400 Bad Request` errors during batch upserts.
- **Successful Validation**:
    - Verified the end-to-end "Boosted" pipeline by successfully parsing, embedding (with TensorRT), and upserting **Cap 282** (440 chunks).

### 🛠️ Current Status
- **Pipeline**: End-to-end GPU-accelerated ingestion and retrieval are fully functional on the new hardware architecture.
- **Chunking**: Semantic paragraph-merging is active, providing significantly better context boundaries for legal text.

### 📅 Next Steps
- [ ] Scale up ingestion to the full ordinance library using the boosted pipeline.
- [ ] Measure and log the final throughput improvement (chunks/sec) compared to the non-boosted baseline.

---
*Log updated on 2026-01-17*

