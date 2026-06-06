"""Minimal Pinecone chunk fetch test — verifies basic connectivity and read operations."""

import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from pinecone import Pinecone


def test_pinecone_fetch():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-rag")

    if not api_key:
        print("ERROR: PINECONE_API_KEY is not set.")
        return False

    print(f"API Key present: {bool(api_key)}")
    print(f"Index name: {index_name}")

    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes().names()
        print(f"Existing indexes: {indexes}")

        if index_name not in indexes:
            print(f"ERROR: Index '{index_name}' does not exist.")
            return False

        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")

        total_vectors = stats.get("total_vector_count", stats.get("totalVectorCount", 0))
        print(f"Total vectors: {total_vectors}")

        if total_vectors == 0:
            print("WARNING: Index has 0 vectors. Nothing to fetch.")
            return False

        # Try a minimal query to get one vector ID
        print("\n--- Querying for a sample vector ---")
        query_response = index.query(
            vector=[0.0] * 1024,
            top_k=1,
            include_metadata=True,
            include_values=False,
        )

        if not query_response.matches:
            print("ERROR: Query returned no matches.")
            return False

        sample_id = query_response.matches[0].id
        print(f"Sample vector ID from query: {sample_id}")

        # Now fetch the actual vector by ID
        print("\n--- Fetching vector by ID ---")
        fetch_response = index.fetch(ids=[sample_id])

        if sample_id not in fetch_response.vectors:
            print(f"ERROR: Fetch returned no vector for ID {sample_id}")
            return False

        vector_data = fetch_response.vectors[sample_id]
        print(f"Fetched vector ID: {sample_id}")
        print(f"Vector dimension: {len(vector_data.values)}")
        print(f"Metadata keys: {list(vector_data.metadata.keys()) if vector_data.metadata else 'None'}")

        print("\n--- SUCCESS: Pinecone fetch test passed ---")
        return True

    except Exception as e:
        print(f"\n--- FAILURE: Pinecone fetch test failed ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        return False


if __name__ == "__main__":
    ok = test_pinecone_fetch()
    sys.exit(0 if ok else 1)
