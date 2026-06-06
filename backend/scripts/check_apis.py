import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
print(f"DeepSeek API Key present: {bool(deepseek_key)}")
if deepseek_key:
    print(f"Key starts with: {deepseek_key[:15]}...")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Say 'DeepSeek OK'"}],
            max_tokens=10
        )
        print(f"DeepSeek Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"DeepSeek Error: {e}")

ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
print(f"\nOllama URL: {ollama_url}")

try:
    import httpx
    r = httpx.get(f"{ollama_url}/api/tags", timeout=10)
    print(f"Ollama Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        models = [m.get('name', m.get('model')) for m in data.get('models', [])]
        print(f"Ollama Models: {models}")
    else:
        print(f"Ollama Response: {r.text[:200]}")
except Exception as e:
    print(f"Ollama Error: {e}")
