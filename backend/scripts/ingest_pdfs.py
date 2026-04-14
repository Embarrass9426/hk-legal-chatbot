import asyncio
import os
import sys
import time
import json
import threading
import uuid
import numpy as np
from queue import Queue, Empty, Full

# Ensure project root is in sys.path (scripts -> backend -> project root = 3 levels)
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.core import setup_env

# Ingestion-specific default: keep larger TRT workspace for embedding throughput.
if os.getenv("EMBEDDING_TRT_MAX_WORKSPACE_SIZE") is None:
    os.environ["EMBEDDING_TRT_MAX_WORKSPACE_SIZE"] = str(6 * 1024**3)

# 1. Setup CUDA/TensorRT environment before any other imports
setup_env.setup_cuda_dlls()

# 2. Clear TensorRT cache before first embedding service usage
from backend.services.embedding_service import EmbeddingService

if os.getenv("EMBEDDING_CLEAR_TRT_CACHE", "0").strip().lower() in {"1", "true", "yes"}:
    EmbeddingService.clear_tensorrt_cache()

from backend.services.vector_store import VectorStoreManager
from backend.services.embedding_service import get_embedding_service
from backend.core.embedding_shared import job_q, STOP_TOKEN
from backend.parsers.pdf_parser import PDFLegalParserV2


# ------------------------------------------------------------------------------
# Worker: Processes embedding requests from the queue
# ------------------------------------------------------------------------------
def embedding_worker():
    """
    Consumes texts from job_q, generates embeddings using the centralized service,
    and puts results back into result_q.
    """
    print("[Worker] Embedding thread started.")
    service = get_embedding_service()

    while True:
        job = job_q.get()
        if job is STOP_TOKEN:
            job_q.task_done()
            break

        if job.get("type") != "embed_request":
            job_q.task_done()
            continue

        batched_jobs = [job]
        total_texts = len(job.get("texts", []))
        stop_requested = False
        drain_deadline = time.time() + WORKER_MICROBATCH_WAIT_SECONDS

        while total_texts < WORKER_MICROBATCH_MAX_TEXTS:
            remaining = drain_deadline - time.time()
            if remaining <= 0:
                break

            try:
                extra_job = job_q.get(timeout=remaining)
            except Empty:
                break

            if extra_job is STOP_TOKEN:
                stop_requested = True
                job_q.task_done()
                break

            if extra_job.get("type") != "embed_request":
                job_q.task_done()
                continue

            batched_jobs.append(extra_job)
            total_texts += len(extra_job.get("texts", []))

        merged_texts = []
        for queued_job in batched_jobs:
            merged_texts.extend(queued_job.get("texts", []))

        source_counts = {}
        for queued_job in batched_jobs:
            source = queued_job.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        print(
            "[Worker] Processing micro-batch "
            f"jobs={len(batched_jobs)} texts={len(merged_texts)} queue_size={job_q.qsize()} "
            f"sources={source_counts}"
        )

        merged_error = None
        merged_vectors = None
        try:

            def _embed_with_fallback(texts_chunk, preferred_size):
                if not texts_chunk:
                    return []

                size = max(WORKER_MIN_EMBED_SUBBATCH_TEXTS, preferred_size)
                while True:
                    try:
                        vectors = []
                        for j in range(0, len(texts_chunk), size):
                            vectors.extend(
                                service.embed_documents(texts_chunk[j : j + size])
                            )
                        return vectors
                    except Exception as embed_exc:
                        if size <= WORKER_MIN_EMBED_SUBBATCH_TEXTS:
                            raise
                        next_size = max(WORKER_MIN_EMBED_SUBBATCH_TEXTS, size // 2)
                        if next_size == size:
                            raise
                        print(
                            "[Worker] Embed sub-batch failed at size "
                            f"{size}; retrying with {next_size}. Error: {str(embed_exc)[:140]}"
                        )
                        size = next_size

            merged_vectors = _embed_with_fallback(
                merged_texts, WORKER_EMBED_SUBBATCH_TEXTS
            )
            if len(merged_vectors) != len(merged_texts):
                raise RuntimeError(
                    "Embedding worker returned mismatched vector count "
                    f"({len(merged_vectors)} vs {len(merged_texts)})"
                )
        except Exception as e:
            merged_error = str(e)

        if merged_error is None and merged_vectors is None:
            merged_error = "Embedding worker produced no vectors"

        offset = 0
        for queued_job in batched_jobs:
            job_id = queued_job.get("id", "unknown")
            reply_q = queued_job.get("reply_q")
            text_count = len(queued_job.get("texts", []))

            if merged_error is not None:
                payload = {"id": job_id, "error": merged_error}
            else:
                vectors_payload = merged_vectors if merged_vectors is not None else []
                payload = {
                    "id": job_id,
                    "vectors": vectors_payload[offset : offset + text_count],
                }
            offset += text_count

            if isinstance(reply_q, Queue):
                try:
                    reply_q.put(payload, timeout=5)
                except Full:
                    print(
                        f"[Worker] Warning: reply queue full for job {str(job_id)[:8]}; requester may have timed out"
                    )

            job_q.task_done()

        if stop_requested:
            break


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def format_elapsed_time(seconds):
    """Formats elapsed seconds into Xm Ys."""
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {int(s)}s"


DOC_RETRIEVAL_PREFIX = "Represent this legal document passage for retrieval: "
EMBEDDING_REPLY_TIMEOUT_SECONDS = float(
    os.getenv("INGEST_EMBEDDING_TIMEOUT_SECONDS", "300")
)
INGEST_INFLIGHT_EMBED_JOBS = max(1, int(os.getenv("INGEST_INFLIGHT_EMBED_JOBS", "3")))
WORKER_MICROBATCH_MAX_TEXTS = max(
    1, int(os.getenv("INGEST_WORKER_MICROBATCH_MAX_TEXTS", "256"))
)
WORKER_MICROBATCH_WAIT_SECONDS = max(
    0.0, float(os.getenv("INGEST_WORKER_MICROBATCH_WAIT_SECONDS", "0.05"))
)
WORKER_EMBED_SUBBATCH_TEXTS = max(
    1, int(os.getenv("INGEST_WORKER_EMBED_SUBBATCH_TEXTS", "32"))
)
WORKER_MIN_EMBED_SUBBATCH_TEXTS = max(
    1, int(os.getenv("INGEST_WORKER_MIN_EMBED_SUBBATCH_TEXTS", "4"))
)


def _validate_vector(vector, expected_dim: int):
    if len(vector) != expected_dim:
        return False, f"dimension={len(vector)} expected={expected_dim}"

    arr = np.asarray(vector, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return False, "contains NaN/Inf"

    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return False, f"near-zero norm={norm}"

    return True, "ok"


def _verify_tensorrt_runtime() -> None:
    service = get_embedding_service()
    service.ensure_loaded()

    model_backend = getattr(service.model, "model", None)
    providers = []
    if model_backend is not None and hasattr(model_backend, "get_providers"):
        providers = model_backend.get_providers()

    print(f"[Config] Active embedding providers: {providers}")

    if "TensorrtExecutionProvider" not in providers:
        raise RuntimeError(
            "TensorRT is required for ingestion but TensorrtExecutionProvider is not active. "
            "If running in WSL, activate .venv-wsl and ensure TensorRT + onnxruntime-gpu "
            "are installed in that environment."
        )


def _count_tensorrt_cache_files(cache_dir: str) -> int:
    if not os.path.isdir(cache_dir):
        return 0

    count = 0
    for entry in os.scandir(cache_dir):
        if entry.is_file():
            count += 1
    return count


def _ensure_tensorrt_cache_built() -> None:
    service = get_embedding_service()
    model_path = getattr(service, "model_path", None)
    if not isinstance(model_path, str) or not model_path:
        print("[TensorRT Cache] Could not resolve model path; skipping cache check")
        return

    cache_dir = os.path.join(model_path, "cache")
    existing_files = _count_tensorrt_cache_files(cache_dir)
    if existing_files > 0:
        print(f"[TensorRT Cache] Built: YES ({existing_files} files)")
        return

    print("[TensorRT Cache] Built: NO (warming up TensorRT cache now)")
    service.ensure_loaded()
    _ = service.embed_query("TensorRT cache warmup probe")

    built_files = _count_tensorrt_cache_files(cache_dir)
    if built_files > 0:
        print(f"[TensorRT Cache] Built: YES ({built_files} files)")
    else:
        print(
            "[TensorRT Cache] Warmup finished but cache file count is still 0; "
            "runtime may be using in-memory engines"
        )


# ------------------------------------------------------------------------------
# Main Ingestion Logic
# ------------------------------------------------------------------------------
async def ingest_legal_pdfs(
    cap_numbers=None,
    batch_size=5,
    layout_batch_size=None,
    embedding_batch_size=128,
    force_parse=False,
    skip_parse=False,
    force_embed=False,
    skip_vector_upload=False,
    wipe_index=False,
    max_caps=None,
):
    """
    Main pipeline:
    1. Scan PDF directory
    2. Start background embedding worker
    3. Parse PDFs (if needed)
    4. Generate embeddings (if needed)
    5. Upsert to Pinecone (unless skipped)
    """

    batch_size = max(1, int(batch_size))
    if layout_batch_size is None:
        layout_batch_size = batch_size
    layout_batch_size = max(1, int(layout_batch_size))

    # 1. Start Background Worker
    worker_thread = threading.Thread(target=embedding_worker, daemon=True)
    worker_thread.start()

    print(
        f"=== Starting Legal PDF Ingestion | concurrency={batch_size}, "
        f"layout_batch={layout_batch_size}, embedding batch={embedding_batch_size} ==="
    )
    print(
        "[Config] TensorRT required="
        f"{os.getenv('EMBEDDING_REQUIRE_TENSORRT', '1')} "
        "| TensorRT FP16="
        f"{os.getenv('EMBEDDING_TRT_FP16', '1')}"
    )
    print(
        "[Config] Worker micro-batch max_texts="
        f"{WORKER_MICROBATCH_MAX_TEXTS} "
        "| wait_seconds="
        f"{WORKER_MICROBATCH_WAIT_SECONDS}"
    )
    print(f"[Config] Worker embed sub-batch texts={WORKER_EMBED_SUBBATCH_TEXTS}")
    print(
        "[Config] Worker minimum embed sub-batch texts="
        f"{WORKER_MIN_EMBED_SUBBATCH_TEXTS}"
    )
    print(f"[Config] Ingest inflight embed jobs per cap={INGEST_INFLIGHT_EMBED_JOBS}")
    start = time.time()
    expected_dim = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_precision = (
        "fp16"
        if os.getenv("EMBEDDING_TRT_FP16", "1").strip().lower()
        not in {"0", "false", "no"}
        else "fp32"
    )
    strict_fp16 = os.getenv("EMBEDDING_STRICT_FP16", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "pdfs")
    parsed_dir = os.path.join(base_dir, "data", "parsed")
    os.makedirs(parsed_dir, exist_ok=True)

    # 2. Identify Target PDFs
    if cap_numbers:
        target_caps = [str(c) for c in cap_numbers]
    else:
        # Scan directory for all PDFs like "cap123.pdf"
        files = os.listdir(pdf_dir)
        target_caps = []
        for f in files:
            if f.lower().startswith("cap") and f.lower().endswith(".pdf"):
                # Extract number part
                part = f[3:-4]  # cap(123).pdf
                if part:
                    target_caps.append(part)

    target_caps = sorted(target_caps, key=lambda x: (len(x), x))

    if max_caps is not None:
        target_caps = target_caps[: max(0, max_caps)]

    print(f"Found {len(target_caps)} ordinances to process: {target_caps}")

    # 3. Initialize Vector Store
    vsm = None
    try:
        _verify_tensorrt_runtime()
        _ensure_tensorrt_cache_built()
    except Exception as e:
        print(f"[Embedding Runtime] TensorRT preflight failed: {e}")
        raise

    if not skip_vector_upload:
        try:
            vsm = VectorStoreManager()
        except Exception as e:
            print(f"[VectorStore] Initialization failed: {e}")
            print("Skipping upload due to VectorStore failure.")
            skip_vector_upload = True

    if wipe_index and not skip_vector_upload and vsm:
        print(
            f"[VectorStore] Deleting all vectors from index '{vsm.index_name}' before ingest..."
        )
        vsm.index.delete(delete_all=True)
        print("[VectorStore] Index wipe completed.")

    # Processing State
    processed = {"count": 0, "total": len(target_caps)}
    processed_lock = threading.Lock()

    def process_cap(cap_num: str):
        try:
            print(f"[Cap {cap_num}] Starting processing")
            json_path = os.path.join(parsed_dir, f"cap{cap_num}.json")
            chunks = []

            if skip_parse and not os.path.exists(json_path):
                print(
                    f"[Skip Parse] Missing parsed JSON for Cap {cap_num}: {json_path}. Skipping."
                )
                return

            if os.path.exists(json_path) and (skip_parse or not force_parse):
                print(
                    f"Found existing parsed JSON for Cap {cap_num}. Skipping parsing."
                )
                parser = PDFLegalParserV2(cap_num, pdf_dir, parsed_dir)
                chunks = parser.load_parsed_json(json_path)
            else:
                print(f"Parsing Cap {cap_num} (force_parse={force_parse})...")
                parser = PDFLegalParserV2(cap_num, pdf_dir, parsed_dir)
                chunks = parser.process_ordinance(
                    skip_if_exists=not force_parse, layout_batch_size=layout_batch_size
                )

            print(f"[Cap {cap_num}] Parsing complete: {len(chunks)} chunks")

            if not chunks:
                print(f"[Warning] No chunks found after parsing Cap {cap_num}.")
                return

            if force_embed:
                print(
                    f"Force mode: Will re-generate embeddings and overwrite for Cap {cap_num}"
                )

            chunk_texts = [DOC_RETRIEVAL_PREFIX + c["content"] for c in chunks]
            total_chunks = len(chunk_texts)
            print(
                f"Generating embeddings for {total_chunks} / {len(chunks)} chunks in Cap {cap_num}..."
            )

            all_vectors = []
            total_batches = (
                total_chunks + embedding_batch_size - 1
            ) // embedding_batch_size
            inflight = []
            next_batch_index = 0

            def _enqueue_batch(batch_index: int):
                start_idx = batch_index * embedding_batch_size
                batch_texts = chunk_texts[start_idx : start_idx + embedding_batch_size]
                batch_num = batch_index + 1
                print(
                    f"Embedding batch {batch_num}/{total_batches} for Cap {cap_num} "
                    f"(size={len(batch_texts)})"
                )

                job_id = str(uuid.uuid4())
                reply_q = Queue(maxsize=1)
                try:
                    job_q.put(
                        {
                            "type": "embed_request",
                            "id": job_id,
                            "texts": batch_texts,
                            "reply_q": reply_q,
                            "source": "ingest",
                        },
                        timeout=EMBEDDING_REPLY_TIMEOUT_SECONDS,
                    )
                except Full as exc:
                    raise RuntimeError(
                        "Embedding worker queue remained full for "
                        f"{EMBEDDING_REPLY_TIMEOUT_SECONDS}s while processing Cap {cap_num}"
                    ) from exc

                inflight.append(
                    {
                        "batch_num": batch_num,
                        "reply_q": reply_q,
                    }
                )

            while (
                next_batch_index < total_batches
                and len(inflight) < INGEST_INFLIGHT_EMBED_JOBS
            ):
                _enqueue_batch(next_batch_index)
                next_batch_index += 1

            while inflight:
                current = inflight.pop(0)
                reply_q = current["reply_q"]
                batch_num = current["batch_num"]

                try:
                    res = reply_q.get(timeout=EMBEDDING_REPLY_TIMEOUT_SECONDS)
                except Empty as exc:
                    raise RuntimeError(
                        "Timed out waiting for embedding worker response for "
                        f"Cap {cap_num} batch {batch_num}/{total_batches} after {EMBEDDING_REPLY_TIMEOUT_SECONDS}s"
                    ) from exc

                if "error" in res:
                    raise RuntimeError(f"Embedding failed: {res['error']}")
                all_vectors.extend(res["vectors"])

                if next_batch_index < total_batches:
                    _enqueue_batch(next_batch_index)
                    next_batch_index += 1

            if skip_vector_upload or not vsm:
                print(
                    f"Skipping upload for Cap {cap_num} (skip_upload={skip_vector_upload})."
                )
            else:
                vectors_to_upsert = []
                invalid_vectors = 0
                for idx, chunk in enumerate(chunks):
                    doc_id = chunk.get("doc_id", f"cap{cap_num}")
                    vector_id = f"{doc_id}_chunk_{idx}"
                    is_valid, reason = _validate_vector(all_vectors[idx], expected_dim)
                    if not is_valid:
                        invalid_vectors += 1
                        print(f"[Vector Validation] Skip {vector_id}: {reason}")
                        continue

                    valid_values = np.asarray(
                        all_vectors[idx], dtype=np.float32
                    ).tolist()
                    metadata = {
                        "content": chunk.get("content", ""),
                        "page_number": chunk.get("page_number", 1),
                        "section_id": chunk.get("section_id", "unknown"),
                        "section_title": chunk.get("section_title", "Unknown Section"),
                        "doc_id": doc_id,
                        "citation": chunk.get("citation", f"Cap. {cap_num}"),
                        "source_url": chunk.get("source_url", ""),
                        "embedding_precision": embedding_precision,
                        "embedding_dimension": expected_dim,
                        "embedding_strict_fp16": strict_fp16,
                    }

                    vectors_to_upsert.append(
                        {
                            "id": vector_id,
                            "values": valid_values,
                            "metadata": metadata,
                        }
                    )

                if invalid_vectors:
                    print(
                        f"[Vector Validation] Cap {cap_num}: filtered {invalid_vectors} invalid vectors before upsert"
                    )

                if not vectors_to_upsert:
                    raise RuntimeError(
                        f"All vectors invalid for Cap {cap_num}; aborting to prevent bad Pinecone writes."
                    )

                upsert_batch_limit = 100
                total_upserted = 0
                for i in range(0, len(vectors_to_upsert), upsert_batch_limit):
                    batch = vectors_to_upsert[i : i + upsert_batch_limit]
                    try:
                        vsm.index.upsert(vectors=batch)
                        total_upserted += len(batch)
                    except Exception as e:
                        print(f"[Error] Upsert failed for batch {i}: {e}")

                print(
                    f"[Worker] Uploaded {total_upserted} vectors for Cap {cap_num} to Pinecone."
                )

            with processed_lock:
                processed["count"] += 1
                done = processed["count"]
            elapsed = format_elapsed_time(time.time() - start)
            print(
                f"Queued Cap {cap_num} embeddings ({len(chunks)} new) | Progress {done}/{processed['total']} | Elapsed {elapsed}"
            )

        except Exception as e:
            print(f"Error processing Cap {cap_num}: {e}")
            import traceback

            traceback.print_exc()
            if "Embedding failed" in str(e):
                print("\n[CRITICAL] Embedding service failed. Stopping ingestion.")
                print("Possible causes:")
                print(
                    "  1. TensorRT cache corruption - try deleting backend/models/yuan-onnx-trt/cache/"
                )
                print("  2. GPU memory exhausted - restart terminal")
                print("  3. CUDA driver issue - check nvidia-smi")
                raise

    semaphore = asyncio.Semaphore(batch_size)

    async def _run_cap(cap_num: str):
        async with semaphore:
            await asyncio.to_thread(process_cap, cap_num)

    await asyncio.gather(*[_run_cap(cap_num) for cap_num in target_caps])

    # Cleanup
    print("Waiting for GPU worker to finish remaining jobs...")
    job_q.put(STOP_TOKEN)
    job_q.join()
    elapsed = format_elapsed_time(time.time() - start)
    print(
        f"\nPipeline complete! Processed {processed['count']} ordinances in {elapsed}."
    )


# ------------------------------------------------------------------------------
# CLI Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest legal PDFs into Pinecone.")
    parser.add_argument("--cap", nargs="+", help="Specific Cap numbers (e.g. 282 599A)")
    parser.add_argument(
        "--batch", type=int, default=5, help="Concurrent parsing jobs (default=5)"
    )
    parser.add_argument(
        "--layout-batch",
        type=int,
        default=None,
        help="PDF parser layout batch size (default=same as --batch)",
    )
    parser.add_argument(
        "--embedding-batch",
        type=int,
        default=128,
        help="Embedding batch size (default=128)",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        default=False,
        help="Force re-parse PDFs even if JSON exists",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        default=False,
        help="Never parse PDFs; only use existing parsed JSON files",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        default=True,
        help="Force re-generate embeddings even if they exist in Pinecone",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of caps to process after filtering",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip Pinecone upsert stage"
    )
    parser.add_argument(
        "--wipe-index",
        action="store_true",
        default=False,
        help="Delete all vectors in Pinecone index before ingesting",
    )
    args = parser.parse_args()

    asyncio.run(
        ingest_legal_pdfs(
            cap_numbers=args.cap,
            batch_size=args.batch,
            layout_batch_size=args.layout_batch,
            embedding_batch_size=args.embedding_batch,
            force_parse=args.force_parse,
            skip_parse=args.skip_parse,
            force_embed=args.force_embed,
            skip_vector_upload=args.skip_upload,
            wipe_index=args.wipe_index,
            max_caps=args.limit,
        )
    )
