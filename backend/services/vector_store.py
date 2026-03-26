import os
import numpy as np
from backend.core import setup_env

setup_env.setup_cuda_dlls()

from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from transformers import AutoTokenizer  # For reranker only
from transformers import AutoModelForSequenceClassification  # For reranker only
import torch
from dotenv import load_dotenv
from backend.services.embedding_service import get_embedding_service

load_dotenv()


class VectorStoreManager:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-rag")

        if not self.api_key:
            print("WARNING: PINECONE_API_KEY not set.")
            return

        self.pc = Pinecone(api_key=self.api_key)
        self.expected_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        self.expected_precision = (
            "fp16"
            if os.getenv("EMBEDDING_TRT_FP16", "1").strip().lower()
            not in {"0", "false", "no"}
            else "fp32"
        )
        self.strict_fp16 = os.getenv(
            "EMBEDDING_STRICT_FP16", "1"
        ).strip().lower() not in {
            "0",
            "false",
            "no",
        }

        # Initialize Embeddings via Singleton Service
        # This will auto-load the model if not already loaded
        print("[VectorStore] connecting to EmbeddingService...")
        self.embeddings = get_embedding_service()

        # Initialize Reranker (Qwen3-Reranker-8B) - DISABLED FOR NOW
        self.reranker_model = None
        self.reranker_tokenizer = None

        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.expected_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

        index_info = self.pc.describe_index(self.index_name)
        actual_dimension = getattr(index_info, "dimension", None)
        if (
            actual_dimension is not None
            and int(actual_dimension) != self.expected_dimension
        ):
            raise ValueError(
                f"Pinecone index dimension mismatch for '{self.index_name}': "
                f"index={actual_dimension}, expected={self.expected_dimension}."
            )

        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.api_key,
        )

    def _validate_query_vector(self, vector: List[float]):
        if len(vector) != self.expected_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: got {len(vector)}, expected {self.expected_dimension}."
            )

        arr = np.asarray(vector, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Query embedding contains NaN/Inf values.")

        l2 = float(np.linalg.norm(arr))
        if l2 <= 1e-8:
            raise ValueError("Query embedding norm is zero/near-zero.")

    def _enforce_precision_regime(self):
        probe = self.index.query(
            vector=[1.0 / self.expected_dimension] * self.expected_dimension,
            top_k=10,
            include_metadata=True,
            include_values=False,
        )

        matches = probe.get("matches", [])
        if not matches:
            return

        for match in matches:
            metadata = match.get("metadata") or {}
            stored_precision = metadata.get("embedding_precision")

            if self.strict_fp16 and not stored_precision:
                raise ValueError(
                    "Strict FP16 mode requires precision metadata on indexed vectors, "
                    "but legacy vectors without embedding_precision were found. "
                    "Re-embed the full corpus in strict FP16 mode."
                )

            if stored_precision and stored_precision != self.expected_precision:
                raise ValueError(
                    "Embedding precision regime mismatch: "
                    f"index sample={stored_precision}, runtime={self.expected_precision}. "
                    "Re-embed corpus with a single precision mode for stable search ranking."
                )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upserts pre-chunked legal data with specific metadata.
        """
        if not self.api_key:
            return

        # Apply asymmetric prefix to chunks
        prefix = "Represent this legal document passage for retrieval: "
        texts = [prefix + c["content"] for c in chunks]
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
                "source_url": c["source_url"],
            }
            metadatas.append(meta)

            # Construct ID and ensure it's under 512 characters
            full_id = f"{c['doc_id']}-{c['section_id']}-{c['chunk_index']}"
            if len(full_id) > 500:
                # Truncate section_id part if too long
                truncated_section = c["section_id"][:400]
                full_id = f"{c['doc_id']}-{truncated_section}-{c['chunk_index']}"
            ids.append(full_id)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.vector_store.add_texts(
                texts=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

    def search_with_expansion(self, query: str, k: int = 10):
        """
        Retrieves top-10 chunks, reranks to top-5, then expands to full section context.
        """
        if not self.api_key:
            return []

        # 1. Similarity search for top-k chunks
        # Apply asymmetric prefix if required by Yuan embedding
        query_with_prefix = (
            f"Represent this question for retrieving relevant legal documents: {query}"
        )
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
                "section_id": {"$eq": section_id},
            }

            section_results = index.query(
                vector=[1.0 / self.expected_dimension] * self.expected_dimension,
                filter=query_filter,
                top_k=100,
                include_metadata=True,
                include_values=False,
            )

            # Extract and sort by chunk_index
            section_chunks = []
            for match in section_results["matches"]:
                section_chunks.append(
                    {
                        "content": match["metadata"].get("text", ""),
                        "metadata": match["metadata"],
                    }
                )

            # Sort by chunk_index
            section_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

            expanded_context.append(
                {"section_id": section_id, "doc_id": doc_id, "chunks": section_chunks}
            )

        return expanded_context

    def search(self, query: str, k: int = 10):
        """
        Pure embedding similarity search (Top-k) without reranking or expansion.
        """
        if not self.api_key:
            return []

        # Apply asymmetric prefix for retrieval query
        query_with_prefix = (
            f"Represent this question for retrieving relevant legal documents: {query}"
        )

        self._enforce_precision_regime()

        query_vector = self.embeddings.embed_query(query_with_prefix)
        self._validate_query_vector(query_vector)

        matches = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            include_values=False,
        )

        from langchain_core.documents import Document

        documents = []
        for match in matches.get("matches", []):
            metadata = match.get("metadata") or {}
            page_content = metadata.get("content", "")
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def _rerank(self, query: str, documents: List[Any], top_k: int = 5):
        """Internal reranking logic using Cross-Encoder model."""
        if not self.reranker_model or not documents:
            return documents[:top_k]

        pairs = [[query, doc.page_content] for doc in documents]

        # Get the device the model is on
        device = next(self.reranker_model.parameters()).device

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            scores = (
                self.reranker_model(**inputs)
                .logits.view(
                    -1,
                )
                .float()
            )

        # Pair scores with documents and sort
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
