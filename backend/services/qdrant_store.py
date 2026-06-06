import os
import re
import uuid
import numpy as np
from typing import List, Dict, Any

from backend.core import setup_env

setup_env.setup_cuda_dlls()

from dotenv import load_dotenv
from langchain_core.documents import Document

from backend.services.embedding_service import get_embedding_service
from backend.services.reranker_service import get_reranker_service

load_dotenv()
load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
)


_ORDINANCE_KEYWORD_RE = re.compile(
    r"^(.+?\s+(?:Ordinance|Regulation|Rules|Act|By-law))",
    re.IGNORECASE,
)

_HYDE_ORDINANCE_RE = re.compile(
    r"\b([A-Z][a-zA-Z\s]*(?:Ordinance|Regulation|Rules|Act|By-law))\b",
    re.IGNORECASE,
)


def extract_ordinance_name_from_title(section_title: str) -> str:
    title = section_title.strip()
    m = _ORDINANCE_KEYWORD_RE.search(title)
    if m:
        return m.group(1).strip()
    words = title.split()
    return " ".join(words[:3]) if len(words) >= 3 else title


def extract_ordinance_names_from_text(text: str) -> List[str]:
    names: set[str] = set()
    for match in _HYDE_ORDINANCE_RE.finditer(text):
        name = match.group(1).strip()
        if len(name) >= 5:
            names.add(name)
    return sorted(names)


def _normalize_for_match(s: str) -> str:
    return "".join(s.lower().split())


def _ordinance_matches(section_title: str, ordinance_names: List[str]) -> bool:
    if not ordinance_names:
        return True
    title_norm = _normalize_for_match(section_title)
    for name in ordinance_names:
        name_norm = _normalize_for_match(name)
        if len(name_norm) >= 3 and title_norm.startswith(name_norm):
            return True
    return False

from qdrant_client import QdrantClient, models


class QdrantStoreManager:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "").strip()
        self.api_key = os.getenv("QDRANT_API_KEY", "").strip()
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "hk_legal_chunks")

        if not self.url:
            print("WARNING: QDRANT_URL not set.")
            return

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key or None,
            prefer_grpc=True,
        )

        self.expected_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        self.expected_precision = (
            "fp16"
            if os.getenv("EMBEDDING_TRT_FP16", "1").strip().lower()
            not in {"0", "false", "no"}
            else "fp32"
        )
        self.strict_fp16 = os.getenv(
            "EMBEDDING_STRICT_FP16", "1"
        ).strip().lower() not in {"0", "false", "no"}

        self.enable_reranker = os.getenv(
            "ENABLE_RERANKER", "1"
        ).strip().lower() not in {"0", "false", "no"}

        self._get_embedding_service = get_embedding_service
        self._get_reranker_service = get_reranker_service
        self._last_rerank_scores: List[Dict[str, Any]] = []

        print("[QdrantStore] EmbeddingService configured for lazy loading")
        if self.enable_reranker:
            print("[QdrantStore] RerankerService configured for lazy loading")
        else:
            print("[QdrantStore] Reranker disabled by ENABLE_RERANKER=0")

        # Create collection if it doesn't exist
        self._ensure_collection()

    @property
    def embeddings(self) -> Any:
        return self._get_embedding_service()

    @property
    def reranker(self) -> Any:
        if not self.enable_reranker:
            return None
        return self._get_reranker_service()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name in existing_names:
            info = self.client.get_collection(self.collection_name)
            actual_dim = info.config.params.vectors.size
            if actual_dim != self.expected_dimension:
                raise ValueError(
                    f"Qdrant collection dimension mismatch for '{self.collection_name}': "
                    f"collection={actual_dim}, expected={self.expected_dimension}."
                )
            return

        print(
            f"[QdrantStore] Creating collection '{self.collection_name}' "
            f"(dim={self.expected_dimension}, distance=cosine)"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.expected_dimension,
                distance=models.Distance.COSINE,
            ),
        )

        for field in ("doc_id", "section_id"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    @staticmethod
    def _to_uuid(id_str: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

    def _search_points(self, query_vector: List[float], limit: int):
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return response.points
        raise RuntimeError(
            "qdrant-client is missing search/query_points methods. "
            "Please upgrade: pip install --upgrade 'qdrant-client>=1.12.0'"
        )

    def delete_all(self):
        if not self.url:
            return
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        if not self.url:
            return
        points = []
        for item in vectors:
            payload = dict(item.get("metadata", {}))
            if "section_title" in payload:
                payload["ordinance_name"] = extract_ordinance_name_from_title(
                    payload["section_title"]
                )
            points.append(
                models.PointStruct(
                    id=self._to_uuid(item["id"]),
                    vector=item["values"],
                    payload=payload,
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
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

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upserts pre-chunked legal data with metadata + ordinance_name payload.
        """
        if not self.url:
            return

        prefix = "Represent this legal document passage for retrieval: "
        texts = [prefix + c["content"] for c in chunks]
        metadatas = []
        ids = []

        for c in chunks:
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
                "content": c["content"],
            }
            # Extract ordinance_name for metadata filtering
            meta["ordinance_name"] = extract_ordinance_name_from_title(c["section_title"])
            metadatas.append(meta)

            full_id = f"{c['doc_id']}-{c['section_id']}-{c['chunk_index']}"
            if len(full_id) > 500:
                truncated = c["section_id"][:400]
                full_id = f"{c['doc_id']}-{truncated}-{c['chunk_index']}"
            ids.append(full_id)

        # Embed in batches
        vectors = self.embeddings.embed_documents(texts)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            points = []
            for j in range(i, min(i + batch_size, len(vectors))):
                points.append(
                    models.PointStruct(
                        id=self._to_uuid(ids[j]),
                        vector=vectors[j],
                        payload=metadatas[j],
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    # ------------------------------------------------------------------
    # Core search primitives
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5):
        """
        Pure embedding similarity search (Top-k) without reranking or expansion.
        """
        if not self.url:
            return []

        query_with_prefix = (
            f"Represent this question for retrieving relevant legal documents: {query}"
        )

        query_vector = self.embeddings.embed_query(query_with_prefix)
        self._validate_query_vector(query_vector)

        results = self._search_points(query_vector, k)

        documents = []
        for scored_point in results:
            payload = scored_point.payload or {}
            page_content = payload.get("content", "")
            documents.append(Document(page_content=page_content, metadata=payload))

        return documents

    def _rerank(self, query: str, documents: List[Any], top_k: int = 5):
        if self.reranker is None:
            return documents[:top_k]
        return self.reranker.rerank(query, documents, top_k=top_k)

    # ------------------------------------------------------------------
    # Expansion helpers
    # ------------------------------------------------------------------

    def _expand_to_sections(
        self, documents: List[Any], max_chunks_per_section: int = 100
    ):
        """
        Given a list of chunk Documents, fetch all sibling chunks for each
        unique (doc_id, section_id) via Qdrant scroll with filter.
        """
        sections_to_fetch = set()
        for doc in documents:
            doc_id = doc.metadata.get("doc_id")
            section_id = doc.metadata.get("section_id")
            if doc_id and section_id:
                sections_to_fetch.add((doc_id, section_id))

        expanded_context = []
        for doc_id, section_id in sections_to_fetch:
            filter_must = [
                models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id)
                ),
                models.FieldCondition(
                    key="section_id", match=models.MatchValue(value=section_id)
                ),
            ]

            # Use scroll to fetch all chunks in the section
            section_chunks = []
            next_offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(must=filter_must),
                    limit=max_chunks_per_section,
                    offset=next_offset,
                    with_payload=True,
                )
                for record in records:
                    payload = record.payload or {}
                    content = payload.get("content", "")
                    section_chunks.append({"content": content, "metadata": payload})

                if next_offset is None or len(section_chunks) >= max_chunks_per_section:
                    break

            section_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            expanded_context.append(
                {"section_id": section_id, "doc_id": doc_id, "chunks": section_chunks}
            )

        return expanded_context

    @staticmethod
    def _flatten_expanded_sections(
        expanded: List[Dict[str, Any]],
    ) -> List[Any]:
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

        grouped: OrderedDict[tuple[str, str], List[Dict[str, Any]]] = OrderedDict()
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

    # ------------------------------------------------------------------
    # Convenience search methods
    # ------------------------------------------------------------------

    def search_with_expansion(self, query: str, k: int = 5):
        """
        Retrieves top-k chunks via similarity search, then expands each to its full section.
        """
        if not self.url:
            return []

        initial_results = self.search(query, k=k)
        return self._expand_to_sections(initial_results)

    def search_with_rerank_and_expansion(
        self, query: str, k: int = 5, rerank_top_k: int = 5
    ):
        """
        Vector search top_k -> rerank to rerank_top_k -> expand to full sections.
        """
        if not self.url:
            return []

        initial_results = self.search(query, k=k)

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
        HyDE pipeline: vector search using pre-computed HyDE embedding -> expand
        to full sections -> rerank using original user query -> return top sections.
        """
        if not self.url:
            return []

        self._validate_query_vector(hyde_embedding)

        results = self._search_points(hyde_embedding, k)

        initial_docs = []
        for scored_point in results:
            payload = scored_point.payload or {}
            page_content = payload.get("content", "")
            initial_docs.append(Document(page_content=page_content, metadata=payload))

        expanded = self._expand_to_sections(
            initial_docs, max_chunks_per_section=max_chunks_per_section
        )
        flat_docs = self._flatten_expanded_sections(expanded)

        if self.reranker is None:
            return self._regroup_to_sections(flat_docs[:rerank_top_k])

        reranked = self.reranker.rerank(original_query, flat_docs, top_k=rerank_top_k)
        return self._regroup_to_sections(reranked)



    def search_multi_hyde_with_rerank_and_expansion(
        self,
        passages: List[str],
        original_query: str,
        k_per_query: int = 100,
        rerank_top_k: int = 5,
        max_chunks_per_section: int = 20,
        capture_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        if not self.url or not passages:
            return []

        if capture_scores:
            self._last_rerank_scores = []

        doc_prefix = "Represent this legal document passage for retrieval: "
        prefixed_texts = [doc_prefix + p for p in passages]
        all_embeddings = self.embeddings.embed_documents(prefixed_texts)

        merged_docs: List[Any] = []
        seen_ids: set[str] = set()

        for passage, embedding in zip(passages, all_embeddings):
            self._validate_query_vector(embedding)

            results = self._search_points(embedding, k_per_query)

            for scored_point in results:
                payload = scored_point.payload or {}
                match_id = str(scored_point.id)

                if match_id in seen_ids:
                    continue

                seen_ids.add(match_id)
                page_content = payload.get("content", "")
                merged_docs.append(
                    Document(page_content=page_content, metadata=payload)
                )

        if not merged_docs:
            return []

        print(
            f"[MultiHyDE] Merged {len(merged_docs)} unique chunks from "
            f"{len(passages)} passages"
        )

        expanded = self._expand_to_sections(
            merged_docs, max_chunks_per_section=max_chunks_per_section
        )
        flat_docs = self._flatten_expanded_sections(expanded)

        if self.reranker is None:
            return self._regroup_to_sections(flat_docs[:rerank_top_k])

        if capture_scores:
            scored = self.reranker.rerank_with_scores(
                original_query, flat_docs, top_k=rerank_top_k
            )
            self._last_rerank_scores = [
                {
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "section_id": doc.metadata.get("section_id", ""),
                    "score": round(score, 4),
                }
                for doc, score in scored
            ]
            reranked = [doc for doc, _ in scored]
        else:
            reranked = self.reranker.rerank(
                original_query, flat_docs, top_k=rerank_top_k
            )

        return self._regroup_to_sections(reranked)
