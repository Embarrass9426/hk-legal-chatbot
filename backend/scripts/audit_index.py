# pyright: reportDeprecated=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

# Ensure project root is in sys.path (scripts -> backend -> project root = 3 levels)
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.core import setup_env

# Must run before importing modules that may pull torch/onnxruntime transitively.
setup_env.setup_cuda_dlls()

from backend.services.vector_store import VectorStoreManager


CORE_CAPS: list[tuple[str, str]] = [
    ("cap57", "Employment Ordinance"),
    ("cap179", "Matrimonial Causes Ordinance"),
    ("cap192", "Matrimonial Proceedings"),
    ("cap200", "Crimes Ordinance (historically renumbered)"),
    ("cap212", "Offences Against the Person"),
    ("cap221", "Criminal Procedure"),
    ("cap232", "Police Force Ordinance"),
    ("cap282", "Employees' Compensation"),
    ("cap338", "Small Claims Tribunal"),
    ("cap347", "Limitation of Actions"),
    ("cap455", "Organized & Serious Crimes"),
]

DOC_ID_PATTERN = re.compile(
    r"^(?P<doc_id>.+?)_chunk_(?P<chunk_idx>\d+)$", re.IGNORECASE
)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_doc_id(vector_id: str) -> str:
    match = DOC_ID_PATTERN.match(vector_id)
    if match:
        return str(match.group("doc_id")).strip().lower()
    marker = "_chunk_"
    lowered = vector_id.lower()
    if marker in lowered:
        return lowered.split(marker, 1)[0].strip()
    return lowered.strip()


def _extract_ids_from_page(page: Any) -> list[str]:
    if page is None:
        return []

    # Common API shape: {"vectors": [{"id": "..."}, ...], "pagination": {...}}
    vectors = _safe_get(page, "vectors", None)
    if vectors is not None:
        ids: list[str] = []
        for item in vectors:
            value = _safe_get(item, "id", None)
            if value:
                ids.append(str(value))
        return ids

    # Alternate shape: {"ids": ["...", ...]}
    ids_field = _safe_get(page, "ids", None)
    if ids_field is not None:
        ids = []
        for item in ids_field:
            if isinstance(item, str):
                ids.append(item)
            else:
                value = _safe_get(item, "id", None)
                if value:
                    ids.append(str(value))
        return ids

    # Iterable of IDs directly
    if isinstance(page, (list, tuple, set)):
        ids = []
        for item in page:
            if isinstance(item, str):
                ids.append(item)
            else:
                value = _safe_get(item, "id", None)
                if value:
                    ids.append(str(value))
        return ids

    return []


def _iterate_vector_ids(index: Any, namespace: str = "") -> Iterator[str]:
    # Prefer explicit pagination if available.
    if hasattr(index, "list_paginated"):
        token: str | None = None
        while True:
            page = index.list_paginated(
                namespace=namespace, limit=99, pagination_token=token
            )
            ids = _extract_ids_from_page(page)
            for vector_id in ids:
                yield vector_id

            pagination = _safe_get(page, "pagination", None)
            next_token = _safe_get(pagination, "next", None)
            if not next_token:
                break
            token = str(next_token)
        return

    if not hasattr(index, "list"):
        raise RuntimeError(
            "Pinecone index client does not expose list/list_paginated APIs."
        )

    # Some client versions return an iterator of pages; others may return one page.
    try:
        listed = index.list(namespace=namespace)
    except TypeError:
        listed = index.list()

    # Single page object (dict-like)
    direct_ids = _extract_ids_from_page(listed)
    if direct_ids:
        for vector_id in direct_ids:
            yield vector_id
        return

    # Iterator/generator of pages or IDs
    if isinstance(listed, Iterable):
        for page in listed:
            page_ids = _extract_ids_from_page(page)
            if page_ids:
                for vector_id in page_ids:
                    yield vector_id
            elif isinstance(page, str):
                yield page


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _extract_vectors_map(fetch_result: Any) -> dict[str, Any]:
    vectors = _safe_get(fetch_result, "vectors", None)
    if isinstance(vectors, dict):
        return vectors
    return {}


def _audit_doc_ids(
    index: Any, namespace: str = ""
) -> tuple[Counter[str], dict[str, set[object]], str | None, int]:
    doc_chunk_counts: Counter[str] = Counter()
    doc_pages: dict[str, set[object]] = defaultdict(set)
    total_ids_scanned = 0

    try:
        all_ids = list(_iterate_vector_ids(index=index, namespace=namespace))
    except Exception as exc:  # noqa: BLE001
        return (
            doc_chunk_counts,
            doc_pages,
            f"Failed to list vector IDs: {exc}",
            total_ids_scanned,
        )

    total_ids_scanned = len(all_ids)
    if total_ids_scanned == 0:
        return doc_chunk_counts, doc_pages, None, total_ids_scanned

    # First pass from IDs
    vector_id_to_doc: dict[str, str] = {}
    for vector_id in all_ids:
        doc_id = _extract_doc_id(vector_id)
        vector_id_to_doc[vector_id] = doc_id
        doc_chunk_counts[doc_id] += 1

    # Optional metadata enrich for page counts and authoritative doc_id.
    # If fetch fails, keep chunk counts derived from IDs.
    try:
        for batch_ids in _batched(all_ids, 200):
            fetch_result = index.fetch(ids=batch_ids, namespace=namespace)
            vectors_map = _extract_vectors_map(fetch_result)
            if not vectors_map:
                continue

            for vector_id, vector_obj in vectors_map.items():
                fallback_doc = vector_id_to_doc.get(
                    vector_id, _extract_doc_id(vector_id)
                )
                metadata = _safe_get(vector_obj, "metadata", {}) or {}
                metadata_doc_id = str(metadata.get("doc_id", "")).strip().lower()
                doc_id = metadata_doc_id if metadata_doc_id else fallback_doc

                if doc_id != fallback_doc:
                    doc_chunk_counts[fallback_doc] -= 1
                    if doc_chunk_counts[fallback_doc] <= 0:
                        del doc_chunk_counts[fallback_doc]
                    doc_chunk_counts[doc_id] += 1
                    vector_id_to_doc[vector_id] = doc_id

                page_number = metadata.get("page_number")
                if page_number is not None:
                    doc_pages[doc_id].add(page_number)
    except Exception as exc:  # noqa: BLE001
        return (
            doc_chunk_counts,
            doc_pages,
            f"Metadata page enrichment failed (chunk counts still valid): {exc}",
            total_ids_scanned,
        )

    return doc_chunk_counts, doc_pages, None, total_ids_scanned


def _format_int(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(value):,}"
    except Exception:  # noqa: BLE001
        return str(value)


def _print_table(rows: list[tuple[str, int, str]]) -> None:
    if not rows:
        print("(no doc_ids discovered)")
        return

    doc_width = max(len("Doc ID"), max(len(row[0]) for row in rows))
    chunks_width = max(len("Chunks"), max(len(_format_int(row[1])) for row in rows))
    pages_width = max(len("Pages"), max(len(row[2]) for row in rows))

    header = f"{'Doc ID':<{doc_width}} | {'Chunks':>{chunks_width}} | {'Pages':>{pages_width}}"
    divider = f"{'-' * doc_width}-|-{'-' * chunks_width}-|-{'-' * pages_width}"
    print(header)
    print(divider)
    for doc_id, chunks, pages in rows:
        print(
            f"{doc_id:<{doc_width}} | {_format_int(chunks):>{chunks_width}} | {pages:>{pages_width}}"
        )


def _normalize_caps(caps: list[str]) -> list[str]:
    normalized: list[str] = []
    for cap in caps:
        token = str(cap).strip().lower()
        if not token:
            continue
        if token.startswith("cap"):
            suffix = token[3:]
            suffix = suffix.strip()
            normalized.append(f"cap{suffix}")
        else:
            normalized.append(f"cap{token}")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Pinecone index coverage by ordinance doc_id/chunks/pages"
    )
    parser.add_argument(
        "--check-caps",
        nargs="*",
        default=[],
        help="Cap numbers (or cap-prefixed tokens) to check, e.g. --check-caps 57 179 cap282",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show top N doc_ids by chunk count (default: show all sorted by doc_id)",
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Pinecone namespace to audit (default: empty/default namespace)",
    )
    args = parser.parse_args()

    vs_manager = VectorStoreManager()
    if not getattr(vs_manager, "api_key", None):
        print("ERROR: PINECONE_API_KEY is not set. Cannot audit index.")
        return 1

    index = vs_manager.index
    stats = index.describe_index_stats()
    total_vector_count = _safe_get(stats, "total_vector_count", None)
    dimension = _safe_get(stats, "dimension", None)

    print("=== Pinecone Index Audit ===")
    print(f"Index: {vs_manager.index_name}")
    print(f"Namespace: {args.namespace if args.namespace else '(default)'}")
    print(f"Total vectors: {_format_int(total_vector_count)}")
    print(f"Dimension: {_format_int(dimension)}")
    print()

    doc_chunk_counts, doc_pages, warning, scanned_ids = _audit_doc_ids(
        index=index,
        namespace=args.namespace,
    )

    if warning:
        print(f"[WARN] {warning}")
        print()

    print(f"Scanned vector IDs: {_format_int(scanned_ids)}")
    print(f"Unique doc_ids discovered: {_format_int(len(doc_chunk_counts))}")
    print()

    if args.top and args.top > 0:
        ordered = sorted(
            doc_chunk_counts.items(), key=lambda item: (-item[1], item[0])
        )[: args.top]
    else:
        ordered = sorted(doc_chunk_counts.items(), key=lambda item: item[0])

    table_rows: list[tuple[str, int, str]] = []
    for doc_id, chunk_count in ordered:
        page_count = len(doc_pages.get(doc_id, set()))
        pages_display = _format_int(page_count) if page_count > 0 else "-"
        table_rows.append((doc_id, chunk_count, pages_display))

    _print_table(table_rows)
    print()

    requested_caps = _normalize_caps(args.check_caps)
    if requested_caps:
        print("=== Requested Cap Checks ===")
        for cap in requested_caps:
            chunks = doc_chunk_counts.get(cap, 0)
            status = "FOUND" if chunks > 0 else "NOT FOUND [!]"
            print(f"{cap:<10} | chunks={_format_int(chunks):>7} | {status}")
        print()

    print("=== Core Ordinance Presence Check ===")
    for cap, label in CORE_CAPS:
        chunks = doc_chunk_counts.get(cap, 0)
        status = "FOUND" if chunks > 0 else "NOT FOUND [!]"
        print(f"{cap:<10} | chunks={_format_int(chunks):>7} | {status} | {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
