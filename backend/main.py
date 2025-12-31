from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import utils
import scraper
import vector_store
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize components
vector_manager = vector_store.VectorStoreManager()
hk_scraper = scraper.HKLIIScraper()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DeepSeek LLM (using OpenAI-compatible interface)
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key or api_key == "your_api_key_here":
    print("WARNING: DEEPSEEK_API_KEY is not set correctly in .env file")

llm = ChatOpenAI(
    model='deepseek-chat', 
    api_key=api_key, 
    base_url='https://api.deepseek.com',
    max_tokens=1024,
    streaming=True
)

class ChatRequest(BaseModel):
    message: str

async def generate_chat_responses(message: str):
    try:
        # 1. Extract Keywords and Target Law
        analysis = utils.extract_keywords(message, llm)
        target_law = analysis.get("target_law")
        print(f"Analysis: {analysis}")

        # 2. Dynamic Scraping (if target law identified)
        # For demo: if user mentions "Cap 1", we scrape it if not already indexed
        if target_law and "Cap" in target_law:
            cap_no = "".join(filter(str.isdigit, target_law))
            if cap_no:
                print(f"Triggering scraper for Cap {cap_no}...")
                sections = hk_scraper.scrape_ordinance(cap_no)
                if sections:
                    print(f"Scraped {len(sections)} sections. Upserting to Pinecone...")
                    vector_manager.upsert_documents(sections)

        # 3. Retrieval from Vector Store
        context_docs = vector_manager.search(message, k=3)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Prepare references for the frontend
        references = []
        for i, doc in enumerate(context_docs):
            ref = doc.metadata
            ref["id"] = f"ref-{i}"
            references.append(ref)

        # 3. Build Augmented Prompt
        system_content = "You are a helpful Hong Kong legal assistant. Provide accurate information based on Hong Kong law."
        if context_text:
            system_content += f"\n\nUse the following legal context to answer the user's question. If the context doesn't contain the answer, say you don't know but provide what information you can from the context. Always cite the specific Ordinance or Case mentioned in the context.\n\nCONTEXT:\n{context_text}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=message)
        ]
        
        # 4. Stream References first (optional, but good for UI)
        if references:
            yield f"data: {json.dumps({'references': references})}\n\n"

        # 5. Stream Answer
        print(f"Starting stream for message: {message}")
        async for chunk in llm.astream(messages):
            if chunk.content:
                data = json.dumps({'answer': chunk.content})
                yield f"data: {data}\n\n"
        print("Stream finished")
    except Exception as e:
        print(f"Error in stream: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return StreamingResponse(generate_chat_responses(request.message), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "HK Legal Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
