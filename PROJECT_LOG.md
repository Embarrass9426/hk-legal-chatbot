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
