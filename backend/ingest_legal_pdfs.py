import asyncio
import os
import glob
import warnings
import time
from threading import Thread, Lock

# ------------------------------------------------------------------------------
# 🧩 Environment setup
# ------------------------------------------------------------------------------
import setup_env
setup_env.setup_cuda_dlls()

os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
os.environ["ORT_TENSORRT_DYNAMIC_SHAPES_ENABLE"] = "0"
os.environ["ORT_TENSORRT_FORCE_SEQUENTIAL"] = "1"
warnings.filterwarnings("ignore", message=".*max_size parameter is deprecated.*")

# ------------------------------------------------------------------------------
# 🧠 Imports
# ------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction

from pdf_parser_v2 import PDFLegalParserV2
from vector_store import VectorStoreManager
from embedding_shared import job_q, result_q, STOP_TOKEN

# ------------------------------------------------------------------------------
# 🔒 Global model state (shared safely)
# ------------------------------------------------------------------------------
MODEL_LOCK = Lock()
SHARED_TOKENIZER = None
SHARED_MODEL = None

# ------------------------------------------------------------------------------
# 🔬 Core embedding functions
# ------------------------------------------------------------------------------
def model_forward(tokenizer, model, texts, device="cuda"):
    """Run the embedding model forward pass safely (GPU-agnostic)."""
    # Replace blank/empty strings to avoid empty attention mask
    texts = [t if t.strip() else "." for t in texts]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Add position IDs if missing
    if "position_ids" not in inputs:
        inputs["position_ids"] = torch.arange(
            0, inputs["input_ids"].size(1),
            dtype=torch.long
        ).unsqueeze(0).expand_as(inputs["input_ids"])

    # Forward pass (supports both PyTorch and ORT)
    if hasattr(model, "model"):  # ONNX/TensorRT or ORTModel
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        outputs = model(**ort_inputs)
        last_hidden = getattr(outputs, "last_hidden_state", outputs[0])
        if isinstance(last_hidden, np.ndarray):
            last_hidden = torch.tensor(last_hidden, dtype=torch.float32)
    else:
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            last_hidden = getattr(outputs, "last_hidden_state", outputs[0])

    # --- Pooling ---
    attn_mask = inputs["attention_mask"].to(last_hidden.device).unsqueeze(-1)
    denom = attn_mask.sum(dim=1).clamp(min=1)  # prevents div by 0
    emb = (last_hidden * attn_mask).sum(dim=1) / denom

    # --- Normalization ---
    emb = F.normalize(emb, p=2, dim=1)

    # --- Final epsilon guard ---
    # Replace NaNs with 0
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    # Detect any all‑zero rows
    zero_rows = (emb.abs().sum(dim=1) == 0)
    if zero_rows.any():
        emb[zero_rows] += torch.rand_like(emb[zero_rows]) * 1e-6  # ensure not all‑zero

    return emb


def safe_embed_texts(texts):
    """Thread-safe embedding call."""
    with MODEL_LOCK:
        return model_forward(SHARED_TOKENIZER, SHARED_MODEL, texts, "cuda")

# ------------------------------------------------------------------------------
# 🧵 Background worker thread
# ------------------------------------------------------------------------------
def embedding_worker():
    """Dedicated worker thread for embeddings and Pinecone upload."""
    print("🚀 [Worker] Embedding thread started.")
    while True:
        item = job_q.get()
        if item is STOP_TOKEN:
            print("🛑 [Worker] Stopping.")
            job_q.task_done()
            break

        # Handle real-time embed requests (parser paragraphs)
        if isinstance(item, dict) and item.get("type") == "embed_request":
            job_id, texts = item["id"], item["texts"]
            try:
                vectors = safe_embed_texts(texts)
                result_q.put({"id": job_id, "vectors": vectors})
            except Exception as e:
                result_q.put({"id": job_id, "error": str(e)})
            finally:
                job_q.task_done()
            continue

        # Handle full document embedding jobs
        cap_num, chunk_batches, chunks, index, skip_upload = item
        try:
            print(f"🎯 [Worker] Embedding Cap {cap_num} ({len(chunk_batches)} batches)")
            all_vectors = []
            for batch_texts in chunk_batches:
                batch_vecs = safe_embed_texts(batch_texts)
                all_vectors.extend(batch_vecs.cpu().numpy())

            # Upload results to Pinecone
            if not skip_upload:
                print(f"📡 [Worker] Uploading {len(all_vectors)} vectors for Cap {cap_num} to Pinecone.")
                upserts = []
                for chunk, vec in zip(chunks, all_vectors):
                    upserts.append((
                        f"{chunk['doc_id']}_chunk_{chunk['chunk_index']}",
                        vec.tolist(),
                        {
                            "doc_id": chunk["doc_id"],
                            "section_id": chunk["section_id"],
                            "section_title": chunk["section_title"],
                            "page_number": chunk["page_number"],
                            "citation": chunk["citation"],
                            "source_url": chunk["source_url"],
                            "content": chunk["content"][:5000],
                        },
                    ))
                # Use batches of 100 for Pinecone
                for i in range(0, len(upserts), 100):
                    index.upsert(vectors=upserts[i:i + 100])
                print(f"✅ [Worker] Uploaded Cap {cap_num} vectors to Pinecone.")
            else:
                print(f"⚡ [Worker] Skipped Pinecone upload for Cap {cap_num}.")

        except Exception as e:
            print(f"❌ [Worker] Error embedding Cap {cap_num}: {e}")
        finally:
            job_q.task_done()


# Start worker thread once
EMBEDDING_THREAD = Thread(target=embedding_worker, daemon=True)
EMBEDDING_THREAD.start()

# ------------------------------------------------------------------------------
# 🧮 Utility
# ------------------------------------------------------------------------------
def format_elapsed_time(seconds):
    hours, rem = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
def get_existing_vector_ids(index, ids):
    """Return the subset of IDs that already exist in the Pinecone index."""
    existing_ids = set()
    try:
        for i in range(0, len(ids), 1000):  # safe small fetch batches
            resp = index.fetch(ids=ids[i : i + 1000])
            existing_ids.update(resp.vectors.keys())
    except Exception as e:
        print(f"⚠️ [FetchExisting] Could not check existing vectors: {e}")
    return existing_ids

# ------------------------------------------------------------------------------
# 📘 Main per-cap ingestion
# ------------------------------------------------------------------------------
async def process_single_cap(
    cap_num,
    pdf_dir,
    parsed_dir,
    semaphore,
    index,
    processed,
    start,
    embedding_batch_size,
    force_reprocess=False,
    skip_upload=False,
):
    async with semaphore:
        try:
            json_path = os.path.join(parsed_dir, f"cap{cap_num}.json")
            parser = PDFLegalParserV2(cap_num, pdf_dir=pdf_dir, output_dir=parsed_dir)

            # ─────────────────────────────────────────────
            # 🧩 1️⃣ PARSING STEP
            # ─────────────────────────────────────────────
            if not force_reprocess and os.path.exists(json_path):
                print(f"⏩ Found existing parsed JSON for Cap {cap_num}. Skipping parsing.")
                # Load existing chunks for embedding step
                chunks = parser.load_parsed_json(json_path)
            else:
                print(f"📖 Parsing Cap {cap_num}...")
                chunks = await asyncio.to_thread(parser.process_ordinance, skip_if_exists=not force_reprocess)
                if not chunks:
                    print(f"⚠️ No chunks found after parsing Cap {cap_num}.")
                    processed["count"] += 1
                    return

            # ─────────────────────────────────────────────
            # 🧩 2️⃣ EMBEDDING CHECK STEP
            # ─────────────────────────────────────────────
            # Build vector IDs (one per chunk)
            vector_ids = [f"{chunk['doc_id']}_chunk_{chunk['chunk_index']}" for chunk in chunks]

            # Query Pinecone to see which ones already exist
            async def fetch_existing():
                return await asyncio.to_thread(get_existing_vector_ids, index, vector_ids)

            existing_ids = await fetch_existing()
            missing_chunks = [chunk for chunk in chunks if f"{chunk['doc_id']}_chunk_{chunk['chunk_index']}" not in existing_ids]

            if not missing_chunks:
                print(f"✅ Cap {cap_num} already has all embeddings in Pinecone ({len(chunks)} vectors). Skipping.")
                processed["count"] += 1
                return

            # ─────────────────────────────────────────────
            # 🧩 3️⃣ EMBEDDING GENERATION STEP
            # ─────────────────────────────────────────────
            if skip_upload:
                print(f"⚡ Skipping upload for Cap {cap_num} (skip_upload=True).")
                processed["count"] += 1
                return

            # Delete old doc entries only if reprocessing
            if force_reprocess:
                await asyncio.to_thread(index.delete, filter={"doc_id": f"cap{cap_num}"})

            print(f"🧠 Generating embeddings for {len(missing_chunks)} / {len(chunks)} chunks in Cap {cap_num}...")

            texts = [f"{c['section_title']}\n{c['content']}" for c in missing_chunks]
            chunk_batches = [texts[i:i + embedding_batch_size] for i in range(0, len(texts), embedding_batch_size)]

            # Queue the embedding job for the worker
            job_q.put((cap_num, chunk_batches, missing_chunks, index, skip_upload))

            processed["count"] += 1
            elapsed = format_elapsed_time(time.time() - start)
            print(f"📦 Queued Cap {cap_num} embeddings ({len(missing_chunks)} new) | Progress {processed['count']}/{processed['total']} | Elapsed {elapsed}")

        except Exception as e:
            print(f"❌ Error processing Cap {cap_num}: {e}")

# ------------------------------------------------------------------------------
# 🚀 Main pipeline entry
# ------------------------------------------------------------------------------
async def ingest_legal_pdfs(
    cap_numbers=None,
    batch_size=10,
    embedding_batch_size=256,
    force_reprocess=False,
    skip_vector_upload=False,
):
    print(f"=== Starting Legal PDF Ingestion | concurrency={batch_size}, embedding batch={embedding_batch_size} ===")

    base_dir = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data"
    pdf_dir = os.path.join(base_dir, "pdfs")
    parsed_dir = os.path.join(base_dir, "parsed")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)

    # Initialize Pinecone
    vsm = VectorStoreManager()
    index = vsm.pc.Index(vsm.index_name)

    # ------------------------------------------------------------------------------
    # Load Yuan model ONCE
    # ------------------------------------------------------------------------------
    global SHARED_MODEL, SHARED_TOKENIZER
    if SHARED_MODEL is None:
        model_path = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\yuan-onnx-trt"
        model_name = "IEITYuan/Yuan-embedding-2.0-en"

        if os.path.exists(os.path.join(model_path, "model.onnx")):
            print("🎯 Loading Yuan ONNX/TensorRT model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            sess_opt = ort.SessionOptions()
            sess_opt.intra_op_num_threads = 1
            sess_opt.inter_op_num_threads = 1

            model = ORTModelForFeatureExtraction.from_pretrained(
                model_path,
                providers=[
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                session_options=sess_opt,
                trust_remote_code=True,
            )
            SHARED_MODEL = model
            SHARED_TOKENIZER = tokenizer
            providers = getattr(model.model, "providers", None)
            print("✅ Model ready with providers:", providers)
        else:
            print("⚠️ Falling back to standard HuggingFace model")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            model.eval()
            SHARED_MODEL, SHARED_TOKENIZER = model, tokenizer

    # ------------------------------------------------------------------------------
    # Cap selection + concurrency setup
    # ------------------------------------------------------------------------------
    if not cap_numbers:
        pdfs = sorted(glob.glob(os.path.join(pdf_dir, "cap*.pdf")))
        cap_numbers = [os.path.basename(p).replace("cap", "").replace(".pdf", "") for p in pdfs]

        import re
        def cap_sort_key(c):
            m = re.match(r"(\d+)([A-Z]*)", c)
            return (int(m.group(1)), m.group(2)) if m else (0, c)
        cap_numbers.sort(key=cap_sort_key)

    processed = {"count": 0, "total": len(cap_numbers)}
    sem = asyncio.Semaphore(batch_size)
    start = time.time()

    tasks = [
        process_single_cap(
            cap, pdf_dir, parsed_dir, sem, index,
            processed, start,
            embedding_batch_size,
            force_reprocess, skip_vector_upload
        )
        for cap in cap_numbers
    ]
    await asyncio.gather(*tasks)

    print("🧭 Waiting for GPU worker to finish remaining jobs…")
    job_q.put(STOP_TOKEN)
    job_q.join()
    elapsed = format_elapsed_time(time.time() - start)
    print(f"\n✅ Pipeline complete! Processed {processed['count']} ordinances in {elapsed}.")

# ------------------------------------------------------------------------------
# 🧭 CLI Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest legal PDFs into Pinecone.")
    parser.add_argument("--cap", nargs="+", help="Specific Cap numbers (e.g. 282 599A)")
    parser.add_argument("--batch", type=int, default=10, help="Concurrent parsing jobs (default=10)")
    parser.add_argument("--embedding-batch", type=int, default=256, help="Embedding batch size (default=256)")
    parser.add_argument("--force", action="store_true", help="Force re-parse PDFs even if JSON exists")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Pinecone upsert stage")
    args = parser.parse_args()

    asyncio.run(
        ingest_legal_pdfs(
            cap_numbers=args.cap,
            batch_size=args.batch,
            embedding_batch_size=args.embedding_batch,
            force_reprocess=args.force,
            skip_vector_upload=args.skip_upload,
        )
    )
