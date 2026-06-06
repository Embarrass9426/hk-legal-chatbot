import os
import sys
import requests
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    print("ERROR: PINECONE_API_KEY not set")
    sys.exit(1)

headers = {"Api-Key": api_key, "Accept": "application/json"}

print("=" * 60)
print("PINECONE ACCOUNT DIAGNOSTIC")
print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
print("=" * 60)

# 1. Check Organization / Project info
print("\n--- 1. Organization Info ---")
try:
    resp = requests.get("https://api.pinecone.io/organizations", headers=headers)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Orgs: {data}")
    else:
        print(f"Body: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# 2. Check Usage
print("\n--- 2. Usage Metrics ---")
try:
    # Pinecone doesn't have a simple usage endpoint, but we can try
    resp = requests.get("https://api.pinecone.io/usage", headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# 3. List indexes with detailed info
print("\n--- 3. Index Details ---")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print(f"Index list response type: {type(indexes)}")
    print(f"Index names: {indexes.names() if hasattr(indexes, 'names') else indexes}")

    for idx_name in (indexes.names() if hasattr(indexes, 'names') else []):
        print(f"\n  Index: {idx_name}")
        try:
            info = pc.describe_index(idx_name)
            print(f"    Dimension: {getattr(info, 'dimension', 'N/A')}")
            print(f"    Metric: {getattr(info, 'metric', 'N/A')}")
            print(f"    Host: {getattr(info, 'host', 'N/A')}")
            print(f"    Spec: {getattr(info, 'spec', 'N/A')}")
            print(f"    Status: {getattr(info, 'status', 'N/A')}")
        except Exception as e:
            print(f"    Error describing index: {e}")
except Exception as e:
    print(f"Error: {e}")

# 4. Try to get index stats (this worked before)
print("\n--- 4. Index Stats (metadata-free ops) ---")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-index")
    if index_name in pc.list_indexes().names():
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Stats: {stats}")
except Exception as e:
    print(f"Error: {e}")

# 5. Try a metadata-only query to see if it's metadata causing the issue
print("\n--- 5. Query without metadata (values only) ---")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-index")
    index = pc.Index(index_name)
    resp = index.query(vector=[0.0]*1024, top_k=1, include_metadata=False, include_values=True)
    print(f"Success! Usage: {getattr(resp, 'usage', 'N/A')}")
    print(f"Matches: {len(resp.matches)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# 6. Check environment for multiple pinecone configs
print("\n--- 6. Environment Check ---")
for k, v in os.environ.items():
    if "pinecone" in k.lower() or "PINECONE" in k:
        masked = v[:8] + "..." if len(v) > 12 else v
        print(f"  {k}={masked}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
