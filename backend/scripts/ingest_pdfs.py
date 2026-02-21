import asyncio
import os
import sys
import time
import json
import threading
import uuid
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path (scripts -> backend -> project root = 3 levels)
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.core import setup_env

# 1. Setup CUDA/TensorRT environment before any other imports
setup_env.setup_cuda_dlls()

# 2. Clear TensorRT cache before first embedding service usage
from backend.services.embedding_service import EmbeddingService

EmbeddingService.clear_tensorrt_cache()

from backend.services.vector_store import VectorStoreManager
from backend.services.embedding_service import get_embedding_service
from backend.core.embedding_shared import job_q, result_q, STOP_TOKEN
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

        if job["type"] == "embed_request":
            texts = job["texts"]
            job_id = job["id"]
            try:
                # Use centralized service for embeddings
                vectors = service.embed_documents(texts)
                result_q.put({"id": job_id, "vectors": vectors})
            except Exception as e:
                result_q.put({"id": job_id, "error": str(e)})
            finally:
                job_q.task_done()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def format_elapsed_time(seconds):
    """Formats elapsed seconds into Xm Ys."""
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {int(s)}s"


# ------------------------------------------------------------------------------
# Main Ingestion Logic
# ------------------------------------------------------------------------------
async def ingest_legal_pdfs(
    cap_numbers=None,
    batch_size=10,
    embedding_batch_size=128,
    force_parse=False,
    force_embed=False,
    skip_vector_upload=False,
):
    """
    Main pipeline:
    1. Scan PDF directory
    2. Start background embedding worker
    3. Parse PDFs (if needed)
    4. Generate embeddings (if needed)
    5. Upsert to Pinecone (unless skipped)
    """

    # 1. Start Background Worker
    worker_thread = threading.Thread(target=embedding_worker, daemon=True)
    worker_thread.start()

    print(
        f"=== Starting Legal PDF Ingestion | concurrency={batch_size}, embedding batch={embedding_batch_size} ==="
    )
    start = time.time()

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
    print(f"Found {len(target_caps)} ordinances to process: {target_caps}")

    # 3. Initialize Vector Store
    vsm = None
    if not skip_vector_upload:
        try:
            vsm = VectorStoreManager()
            # Initialize embeddings explicitly to trigger model loading early
            get_embedding_service()
        except Exception as e:
            print(f"[VectorStore] Initialization failed: {e}")
            print("Skipping upload due to VectorStore failure.")
            skip_vector_upload = True

    # Processing State
    processed = {"count": 0, "total": len(target_caps)}

    # Process each ordinance
    for cap_num in target_caps:
        try:
            # A. Parse (or Load) JSON
            json_path = os.path.join(parsed_dir, f"cap{cap_num}.json")
            chunks = []

            if os.path.exists(json_path) and not force_parse:
                print(
                    f"Found existing parsed JSON for Cap {cap_num}. Skipping parsing."
                )
                parser = PDFLegalParserV2(cap_num, pdf_dir, parsed_dir)
                chunks = parser.load_parsed_json(json_path)
            else:
                print(f"Parsing Cap {cap_num} (force_parse={force_parse})...")
                parser = PDFLegalParserV2(cap_num, pdf_dir, parsed_dir)
                chunks = parser.process_ordinance(
                    skip_if_exists=not force_parse, layout_batch_size=batch_size
                )

            if not chunks:
                print(f"[Warning] No chunks found after parsing Cap {cap_num}.")
                continue

            if force_embed:
                print(
                    f"Force mode: Will re-generate embeddings and overwrite for Cap {cap_num}"
                )
            else:
                # Simple check if embeddings exist (mock check for now to avoid Pinecone latency)
                pass

            # Prepare for Embedding
            chunk_texts = [c["content"] for c in chunks]
            total_chunks = len(chunk_texts)
            print(
                f"Generating embeddings for {total_chunks} / {len(chunks)} chunks in Cap {cap_num}..."
            )

            all_vectors = []
            for i in range(0, total_chunks, embedding_batch_size):
                batch_texts = chunk_texts[i : i + embedding_batch_size]

                job_id = str(uuid.uuid4())
                job_q.put({"type": "embed_request", "id": job_id, "texts": batch_texts})

                while True:
                    res = result_q.get()
                    if res.get("id") == job_id:
                        if "error" in res:
                            raise RuntimeError(f"Embedding failed: {res['error']}")
                        all_vectors.extend(res["vectors"])
                        break
                    else:
                        result_q.put(res)
                        time.sleep(0.01)

            # D. Upsert to Pinecone
            if skip_vector_upload or not vsm:
                print(
                    f"Skipping upload for Cap {cap_num} (skip_upload={skip_vector_upload})."
                )
            else:
                vectors_to_upsert = []
                for idx, chunk in enumerate(chunks):
                    # Robust ID generation
                    doc_id = chunk.get("doc_id", f"cap{cap_num}")
                    chunk_idx = chunk.get("chunk_index", idx + 1)
                    vector_id = f"{doc_id}_chunk_{chunk_idx}"

                    # Ensure metadata fields exist
                    metadata = {
                        "content": chunk.get("content", ""),
                        "page_number": chunk.get("page_number", 1),
                        "section_id": chunk.get("section_id", "unknown"),
                        "section_title": chunk.get("section_title", "Unknown Section"),
                        "doc_id": doc_id,
                        "citation": chunk.get("citation", f"Cap. {cap_num}"),
                        "source_url": chunk.get("source_url", ""),
                    }

                    vectors_to_upsert.append(
                        {
                            "id": vector_id,
                            "values": all_vectors[idx],
                            "metadata": metadata,
                        }
                    )

                # Batch upsert
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

            processed["count"] += 1
            elapsed = format_elapsed_time(time.time() - start)
            print(
                f"Queued Cap {cap_num} embeddings ({len(chunks)} new) | Progress {processed['count']}/{processed['total']} | Elapsed {elapsed}"
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
                sys.exit(1)

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
        "--batch", type=int, default=10, help="Concurrent parsing jobs (default=10)"
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
        "--force-embed",
        action="store_true",
        default=True,
        help="Force re-generate embeddings even if they exist in Pinecone",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip Pinecone upsert stage"
    )
    args = parser.parse_args()

    asyncio.run(
        ingest_legal_pdfs(
            cap_numbers=args.cap,
            batch_size=args.batch,
            embedding_batch_size=args.embedding_batch,
            force_parse=args.force_parse,
            force_embed=args.force_embed,
            skip_vector_upload=args.skip_upload,
        )
    )
