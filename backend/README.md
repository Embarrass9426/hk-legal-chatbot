# HK Legal Chatbot Backend

This is the FastAPI backend for the Hong Kong Legal RAG system.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file in this directory and add your DeepSeek API key:
   ```env
   DEEPSEEK_API_KEY=your_api_key_here
   ```

3. **Run the Server**:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000`.

## API Endpoints

- `GET /`: Health check.
- `POST /chat`: Send a message to the chatbot (Streaming).
  - Request Body: `{"message": "your question"}`
  - Response: `text/event-stream` with data chunks: `data: {"answer": "..."}`
