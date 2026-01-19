import os
import setup_env
setup_env.setup_cuda_dlls()

from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

class BoostedYuanEmbeddings:
    """Custom LangChain compatible embedding class using ONNX/TensorRT."""
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            provider="TensorrtExecutionProvider",
            provider_options={
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": model_path
            }
        )
        self.device = "cpu"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Yuan embedding typically expects a prefix for better performance
        # but handled here by the caller if needed.
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Generate position_ids if required by the ORT model
        if "position_ids" not in inputs:
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Yuan-embedding-2.0-en uses the CLS token (index 0) for embeddings
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                # Fallback for dictionary/tuple output
                embeddings = outputs[0][:, 0, :]
                
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().tolist()

class VectorStoreManager:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-rag")
        
        if not self.api_key:
            print("WARNING: PINECONE_API_KEY not set.")
            return

        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize Boosted Embeddings with Yuan-embedding-2.0-en
        model_path = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\yuan-onnx-trt"
        if os.path.exists(os.path.join(model_path, "model.onnx")):
            print("Using Boosted Yuan Embeddings (ONNX/TensorRT)")
            self.embeddings = BoostedYuanEmbeddings(model_path)
        else:
            print("Warning: Boosted model not found. Using standard HuggingFaceEmbeddings.")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="IEITYuan/Yuan-embedding-2.0-en",
                model_kwargs={'device': 'cuda', 'trust_remote_code': True} 
            )
        
        # Initialize Reranker (Qwen3-Reranker-8B) - DISABLED FOR NOW
        self.reranker_model = None
        self.reranker_tokenizer = None
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1024, # Dimension for Yuan-embedding-2.0-en
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.api_key
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upserts pre-chunked legal data with specific metadata.
        """
        if not self.api_key:
            return
            
        # Apply asymmetric prefix to chunks
        prefix = "Represent this legal document passage for retrieval: "
        texts = [prefix + c['content'] for c in chunks]
        metadatas = []
        ids = []
        
        for c in chunks:
            # Metadata as per instruction
            meta = {
                "doc_id": c["doc_id"],
                "section_id": c["section_id"],
                "section_title": c["section_title"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "total_chunks_in_section": c.get("total_chunks_in_section", 1),
                "citation": c["citation"],
                "source_url": c["source_url"]
            }
            metadatas.append(meta)
            
            # Construct ID and ensure it's under 512 characters
            full_id = f"{c['doc_id']}-{c['section_id']}-{c['chunk_index']}"
            if len(full_id) > 500:
                # Truncate section_id part if too long
                truncated_section = c['section_id'][:400]
                full_id = f"{c['doc_id']}-{truncated_section}-{c['chunk_index']}"
            ids.append(full_id)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.vector_store.add_texts(
                texts=texts[i:i + batch_size], 
                metadatas=metadatas[i:i + batch_size], 
                ids=ids[i:i + batch_size]
            )

    def search_with_expansion(self, query: str, k: int = 5):
        """
        Retrieves top-10 chunks, reranks to top-5, then expands to full section context.
        """
        if not self.api_key:
            return []
            
        # 1. Similarity search for top-k chunks
        # Apply asymmetric prefix if required by Yuan embedding
        query_with_prefix = f"Represent this question for retrieving relevant legal documents: {query}"
        # We fetch top-k directly since reranking is disabled
        initial_results = self.vector_store.similarity_search(query_with_prefix, k=k)
        
        # 2. Reranking (DISABLED FOR NOW) - Using similarity results directly
        reranked_results = initial_results
        
        # 3. Identify unique sections from retrieved chunks to expand logic
        sections_to_fetch = set()
        for doc in reranked_results:
            doc_id = doc.metadata.get("doc_id")
            section_id = doc.metadata.get("section_id")
            if doc_id and section_id:
                sections_to_fetch.add((doc_id, section_id))
        
        # 4. Fetch all siblings for each identified section
        expanded_context = []
        index = self.pc.Index(self.index_name)
        
        for doc_id, section_id in sections_to_fetch:
            # Query Pinecone by metadata filter to get all chunks in section
            query_filter = {
                "doc_id": {"$eq": doc_id},
                "section_id": {"$eq": section_id}
            }
            
            section_results = index.query(
                vector=[0]*1024, # Dummy vector for metadata-only filtering
                filter=query_filter,
                top_k=100, 
                include_metadata=True,
                include_values=False
            )
            
            # Extract and sort by chunk_index
            section_chunks = []
            for match in section_results['matches']:
                section_chunks.append({
                    "content": match['metadata'].get('text', ''), 
                    "metadata": match['metadata']
                })
            
            # Sort by chunk_index
            section_chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
            
            expanded_context.append({
                "section_id": section_id,
                "doc_id": doc_id,
                "chunks": section_chunks
            })
            
        return expanded_context

    def search(self, query: str, k: int = 5):
        """
        Pure embedding similarity search (Top-k) without reranking or expansion.
        """
        if not self.api_key:
            return []
            
        # Apply asymmetric prefix for retrieval query
        query_with_prefix = f"Represent this question for retrieving relevant legal documents: {query}"
        
        # Similarity search for top-k chunks
        return self.vector_store.similarity_search(query_with_prefix, k=k)

    def _rerank(self, query: str, documents: List[Any], top_k: int = 5):
        """Internal reranking logic using Cross-Encoder model."""
        if not self.reranker_model or not documents:
            return documents[:top_k]
            
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get the device the model is on
        device = next(self.reranker_model.parameters()).device
        
        with torch.no_grad():
            inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            scores = self.reranker_model(**inputs).logits.view(-1,).float()
            
        # Pair scores with documents and sort
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
