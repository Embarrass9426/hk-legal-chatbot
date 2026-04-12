import os
import numpy as np
from backend.core import setup_env

setup_env.setup_cuda_dlls()

from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from backend.services.embedding_service import get_embedding_service
from backend.services.reranker_service import get_reranker_service

load_dotenv()
load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
)


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
        self.legacy_precision_policy = (
            os.getenv("EMBEDDING_LEGACY_PRECISION_POLICY", "warn").strip().lower()
        )
        if self.legacy_precision_policy not in {"warn", "error"}:
            print(
                "[VectorStore] Invalid EMBEDDING_LEGACY_PRECISION_POLICY="
                f"{self.legacy_precision_policy!r}; defaulting to 'warn'"
            )
            self.legacy_precision_policy = "warn"
        self._precision_regime_checked = False
        self.enable_reranker = os.getenv(
            "ENABLE_RERANKER", "1"
        ).strip().lower() not in {
            "0",
            "false",
            "no",
        }

        self._get_embedding_service = get_embedding_service
        self._get_reranker_service = get_reranker_service

        print("[VectorStore] EmbeddingService configured for lazy loading")
        if self.enable_reranker:
            print("[VectorStore] RerankerService configured for lazy loading")
        else:
            print("[VectorStore] Reranker disabled by ENABLE_RERANKER=0")

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
            embedding=self._get_embedding_service(),  # pyright: ignore[reportArgumentType]
            pinecone_api_key=self.api_key,
            text_key="content",
        )

    @property
    def embeddings(self) -> Any:
        return self._get_embedding_service()

    @property
    def reranker(self) -> Any:
        if not self.enable_reranker:
            return None
        return self._get_reranker_service()

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
        if self._precision_regime_checked:
            return

        probe: Any = self.index.query(
            vector=[1.0 / self.expected_dimension] * self.expected_dimension,
            top_k=10,
            include_metadata=True,
            include_values=False,
        )

        matches = probe.get("matches", [])
        if not matches:
            self._precision_regime_checked = True
            return

        missing_precision_count = 0
        for match in matches:
            metadata = match.get("metadata") or {}
            stored_precision = metadata.get("embedding_precision")

            if not stored_precision:
                missing_precision_count += 1
                continue

            if stored_precision and stored_precision != self.expected_precision:
                raise ValueError(
                    "Embedding precision regime mismatch: "
                    f"index sample={stored_precision}, runtime={self.expected_precision}. "
                    "Re-embed corpus with a single precision mode for stable search ranking."
                )

        if missing_precision_count > 0:
            warning_message = (
                "Detected legacy vectors without embedding_precision metadata "
                f"in query probe sample ({missing_precision_count}/{len(matches)})."
            )
            if self.legacy_precision_policy == "error":
                raise ValueError(
                    "Strict FP16 mode requires precision metadata on indexed vectors, "
                    "but legacy vectors without embedding_precision were found. "
                    "Re-embed the full corpus in strict FP16 mode, or temporarily set "
                    "EMBEDDING_LEGACY_PRECISION_POLICY=warn to allow legacy reads. "
                    + warning_message
                )

            print(
                "[VectorStore] WARNING: "
                + warning_message
                + " Proceeding with runtime precision assumptions for legacy records. "
                + "Recommended action: re-embed full corpus with embedding_precision metadata."
            )

        self._precision_regime_checked = True

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
                "embedding_precision": self.expected_precision,
                "embedding_dimension": self.expected_dimension,
                "embedding_strict_fp16": self.strict_fp16,
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

    def search_with_expansion(self, query: str, k: int = 5):
        """
        Retrieves top-k chunks via similarity search, then expands each to its full section.
        No reranking — just search + expand.
        """
        if not self.api_key:
            return []

        query_with_prefix = (
            f"Represent this question for retrieving relevant legal documents: {query}"
        )
        self._enforce_precision_regime()
        initial_results = self.vector_store.similarity_search(query_with_prefix, k=k)

        return self._expand_to_sections(initial_results)

    def search_with_rerank_and_expansion(
        self, query: str, k: int = 5, rerank_top_k: int = 5
    ):
        """
        Legacy strategy: vector search top_k -> rerank to rerank_top_k -> expand to full sections.
        Kept for backward compatibility with eval framework.
        """
        if not self.api_key:
            return []

        query_with_prefix = (
            f"Represent this question for retrieving relevant legal documents: {query}"
        )
        self._enforce_precision_regime()
        initial_results = self.vector_store.similarity_search(query_with_prefix, k=k)

        if self.reranker is None:
            return self._expand_to_sections(initial_results[:rerank_top_k])

        reranked = self.reranker.rerank(query, initial_results, top_k=rerank_top_k)

        return self._expand_to_sections(reranked)

    def search_hyde_with_rerank_and_expansion(
        self,
        hyde_embedding: List[float],
        original_query: str,
        k: int = 50,
        rerank_top_k: int = 10,
        max_chunks_per_section: int = 20,
    ):
        """
        HyDE pipeline: vector search using pre-computed HyDE embedding → expand
        to full sections → rerank using original user query → return top sections.
        """
        if not self.api_key:
            return []

        self._enforce_precision_regime()
        self._validate_query_vector(hyde_embedding)

        matches: Any = self.index.query(
            vector=hyde_embedding,
            top_k=k,
            include_metadata=True,
            include_values=False,
        )

        from langchain_core.documents import Document

        initial_docs = []
        for match in matches.get("matches", []):
            metadata = match.get("metadata") or {}
            page_content = metadata.get("content", "")
            initial_docs.append(Document(page_content=page_content, metadata=metadata))

        expanded = self._expand_to_sections(
            initial_docs, max_chunks_per_section=max_chunks_per_section
        )

        flat_docs = self._flatten_expanded_sections(expanded)

        if self.reranker is None:
            return self._regroup_to_sections(flat_docs[:rerank_top_k])

        reranked = self.reranker.rerank(original_query, flat_docs, top_k=rerank_top_k)

        return self._regroup_to_sections(reranked)

    def _expand_to_sections(
        self, documents: List[Any], max_chunks_per_section: int = 100
    ):
        """
        Given a list of chunk Documents, fetch all sibling chunks for each unique section.
        """
        sections_to_fetch = set()
        for doc in documents:
            doc_id = doc.metadata.get("doc_id")
            section_id = doc.metadata.get("section_id")
            if doc_id and section_id:
                sections_to_fetch.add((doc_id, section_id))

        expanded_context = []
        index = self.pc.Index(self.index_name)

        for doc_id, section_id in sections_to_fetch:
            query_filter: Any = {
                "doc_id": {"$eq": doc_id},
                "section_id": {"$eq": section_id},
            }

            section_results: Any = index.query(
                vector=[1.0 / self.expected_dimension] * self.expected_dimension,
                filter=query_filter,
                top_k=max_chunks_per_section,
                include_metadata=True,
                include_values=False,
            )

            section_chunks = []
            for match in section_results["matches"]:
                metadata = match["metadata"]
                content = metadata.get("text", "") or metadata.get("content", "")
                section_chunks.append(
                    {
                        "content": content,
                        "metadata": metadata,
                    }
                )

            section_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

            expanded_context.append(
                {"section_id": section_id, "doc_id": doc_id, "chunks": section_chunks}
            )

        return expanded_context

    @staticmethod
    def _flatten_expanded_sections(
        expanded: List[Dict[str, Any]],
    ) -> List[Any]:
        from langchain_core.documents import Document

        flat: List[Any] = []
        for section in expanded:
            for chunk in section.get("chunks", []):
                metadata = chunk.get("metadata", {})
                content = chunk.get("content", "")
                flat.append(Document(page_content=content, metadata=metadata))
        return flat

    @staticmethod
    def _regroup_to_sections(
        documents: List[Any],
    ) -> List[Dict[str, Any]]:
        from collections import OrderedDict

        from typing import Tuple

        grouped: OrderedDict[Tuple[str, str], List[Dict[str, Any]]] = OrderedDict()
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = metadata.get("doc_id", "")
            section_id = metadata.get("section_id", "")
            key = (doc_id, section_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(
                {
                    "content": getattr(doc, "page_content", ""),
                    "metadata": metadata,
                }
            )

        result = []
        for (doc_id, section_id), chunks in grouped.items():
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            result.append(
                {"section_id": section_id, "doc_id": doc_id, "chunks": chunks}
            )
        return result

    def search(self, query: str, k: int = 5):
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

        matches: Any = self.index.query(
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
        if self.reranker is None:
            return documents[:top_k]
        return self.reranker.rerank(query, documents, top_k=top_k)

    def search_multi_hyde_with_rerank_and_expansion(
        self,
        passages: List[str],
        original_query: str,
        k_per_query: int = 3,
        rerank_top_k: int = 5,
        max_chunks_per_section: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Multi-query HyDE pipeline:
        1. Batch-embed all passages with doc-side prefix
        2. Run one Pinecone query per embedding (k_per_query each)
        3. Merge raw hits, deduplicate by (doc_id, section_id)
        4. Expand unique sections once
        5. Flatten → rerank with original query → regroup to sections
        """
        if not self.api_key or not passages:
            return []

        self._enforce_precision_regime()

        doc_prefix = "Represent this legal document passage for retrieval: "
        prefixed_texts = [doc_prefix + p for p in passages]
        all_embeddings = self.embeddings.embed_documents(prefixed_texts)

        from langchain_core.documents import Document

        merged_docs: List[Any] = []
        seen_ids: set[str] = set()

        for embedding in all_embeddings:
            self._validate_query_vector(embedding)
            matches: Any = self.index.query(
                vector=embedding,
                top_k=k_per_query,
                include_metadata=True,
                include_values=False,
            )

            for match in matches.get("matches", []):
                match_id = match.get("id", "")
                if match_id in seen_ids:
                    continue
                seen_ids.add(match_id)

                metadata = match.get("metadata") or {}
                page_content = metadata.get("content", "")
                merged_docs.append(
                    Document(page_content=page_content, metadata=metadata)
                )

        if not merged_docs:
            return []

        expanded = self._expand_to_sections(
            merged_docs, max_chunks_per_section=max_chunks_per_section
        )

        flat_docs = self._flatten_expanded_sections(expanded)

        if self.reranker is None:
            return self._regroup_to_sections(flat_docs[:rerank_top_k])

        reranked = self.reranker.rerank(original_query, flat_docs, top_k=rerank_top_k)
        return self._regroup_to_sections(reranked)
