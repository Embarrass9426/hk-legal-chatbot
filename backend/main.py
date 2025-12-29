from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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
        messages = [
            SystemMessage(content="You are a helpful Hong Kong legal assistant. Provide accurate information based on Hong Kong law."),
            HumanMessage(content=message)
        ]
        
        print(f"Starting stream for message: {message}")
        async for chunk in llm.astream(messages):
            if chunk.content:
                # Send the content chunk as a JSON string
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
