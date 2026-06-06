import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env
setup_env.setup_cuda_dlls()

from backend.services.qdrant_store import QdrantStoreManager

vm = QdrantStoreManager()
print(f"Index: {vm.index_name}")
print(f"Dimension: {vm.expected_dimension}")

query = "What are the requirements for divorce in Hong Kong?"
print(f"\nTesting query: {query}")
results = vm.search(query, k=5)
print(f"Retrieved {len(results)} chunks")
for i, r in enumerate(results[:3]):
    print(f"\n{i+1}. {r.metadata.get('citation', 'N/A')}")
    print(f"   {r.page_content[:200]}...")
