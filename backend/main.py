from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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
    language: str = "en"

async def generate_chat_responses(message: str, language: str = "en"):
    try:
        # 1. Extract Keywords and Target Law
        analysis = await utils.extract_keywords(message, llm)
        print(f"Analysis: {analysis}")

        # 2. Retrieval from Vector Store - Pure Similarity Search (Top-5)
        search_results = vector_manager.search(message, k=5)
        
        # Prepare context text and references
        context_parts = []
        references = []
        
        for i, doc in enumerate(search_results):
            content = doc.page_content
            # Strip retrieval prefix if present
            prefix = "Represent this legal document passage for retrieval: "
            if content.startswith(prefix):
                content = content[len(prefix):]
                
            metadata = doc.metadata
            sec_title = metadata.get("section_title", "Unknown Section")
            citation = metadata.get("citation", f"Section {metadata.get('section_id', 'Unknown')}")
            
            context_parts.append(f"### Reference [{i+1}]: {sec_title}\n{content}")
            
            references.append({
                "id": f"ref-{i}",
                "title": citation,
                "source_url": metadata.get("source_url", ""),
                "page": metadata.get("page_number", ""),
                "jurisdiction": "Hong Kong",
                "type": "Ordinance"
            })

        context_text = "\n\n".join(context_parts)

        # 3. Build Augmented Prompt
        system_content = """You are an expert Hong Kong legal assistant specializing in the Employees' Compensation Ordinance (Cap. 282).
Your goal is to help employees understand their rights regarding workplace injuries and insurance.

Instructions:
1. Use the provided legal context to answer the user's question.
2. If the context doesn't contain the answer, state that you don't have enough information but provide general guidance based on the context.
3. Always cite the specific Section (e.g., [1] Cap. 282, s. 5) when referring to the law.
4. Be empathetic but professional.
5. If the user asks about a specific injury (like breaking a leg), explain if it's covered under "arising out of and in the course of employment".
"""
        
        if language == "zh":
            system_content += "\nIMPORTANT: You MUST respond in Traditional Chinese (繁體中文). Even if the context is in English, translate the relevant parts to help the user understand, but keep the legal citations (e.g. Cap. 282, s. 5) recognizable."
        else:
            system_content += "\nIMPORTANT: Respond in English."

        system_content += "\n\nCONTEXT:\n" + context_text

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=message)
        ]
        
        # 4. Stream Answer
        print(f"Starting stream for message: {message}")
        async for chunk in llm.astream(messages):
            if chunk.content:
                data = json.dumps({'answer': chunk.content})
                yield f"data: {data}\n\n"

        # 5. Stream References after the answer
        if references:
            yield f"data: {json.dumps({'references': references})}\n\n"
        
        print("Stream finished")
    except Exception as e:
        print(f"Error in stream: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return StreamingResponse(generate_chat_responses(request.message, request.language), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "HK Legal Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
