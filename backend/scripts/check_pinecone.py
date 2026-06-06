import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-rag")

print(f"API Key present: {bool(api_key)}")
print(f"Index name: {index_name}")

try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes().names()
    print(f"Existing indexes: {indexes}")
    
    if index_name in indexes:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        total_vectors = stats.get('total_vector_count', stats.get('totalVectorCount', 0))
        print(f"Total vectors: {total_vectors}")
    else:
        print(f"Index '{index_name}' does not exist.")
except Exception as e:
    print(f"Error: {e}")
