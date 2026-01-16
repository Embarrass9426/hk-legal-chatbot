import asyncio
import os
import site
import glob
import warnings

# Fix for missing TensorRT and CUDA DLLs
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib_path)
            
    import tensorrt as trt
    trt_path = os.path.join(os.path.dirname(trt.__file__), "..", "tensorrt_libs")
    if os.path.exists(trt_path):
        os.environ["PATH"] = trt_path + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(trt_path)
except Exception:
    pass

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*max_size parameter is deprecated.*")

from pdf_parser_v2 import PDFLegalParserV2
from vector_store import VectorStoreManager
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch
import torch.nn.functional as F

async def process_single_cap(cap_num, pdf_dir, parsed_dir, semaphore, index, model, tokenizer, processed_counter, embedding_batch_size=16, force_reprocess=False, skip_vector_upload=False):
    """Processes a single ordinance cap with batch embedding and enriched metadata."""
    async with semaphore:
        try:
            parser = PDFLegalParserV2(cap_num, pdf_dir=pdf_dir, output_dir=parsed_dir, model=model, tokenizer=tokenizer)
            json_path = os.path.join(parser.output_dir, f"cap{cap_num}.json")

            
            # Skip if JSON already exists (parsing and embedding)
            if not force_reprocess and os.path.exists(json_path):
                processed_counter["count"] += 1
                print(f"--- Skipping Cap {cap_num} (JSON found) | Progress: {processed_counter['count']}/{processed_counter['total']} ---")
                return

            print(f"\n--- Starting Cap {cap_num} ---")

            # 1. Parse PDF (Heavy CPU/GPU task)
            # Pass force_reprocess down to ensure the parser doesn't just load from disk
            chunks = await asyncio.to_thread(parser.process_ordinance, skip_if_exists=not force_reprocess)
            if not chunks:
                return

            # 2. Clear old embeddings from Pinecone
            if not skip_vector_upload:
                await asyncio.to_thread(index.delete, filter={"doc_id": f"cap{cap_num}"})

            # 3. Batch Embed semantic text (section title + content)
            print(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [
                f"{c['section_title']}\n{c['content']}"
                for c in chunks
            ]

            # Force CPU for torch tensors to avoid sm_120 compatibility issues 
            # ORT will handle GPU execution via TensorRT/CUDA providers internally
            device = "cpu"
            vectors = []
            
            # Use unified manual batching
            for i in range(0, len(texts), embedding_batch_size):
                batch_texts = texts[i:i + embedding_batch_size]
                with torch.no_grad():
                    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
                    
                    # Generate position_ids if required by the ORT model
                    if "position_ids" not in inputs:
                        b_size, s_len = inputs["input_ids"].shape
                        inputs["position_ids"] = torch.arange(s_len, device=device).unsqueeze(0).expand(b_size, -1)

                    outputs = model(**inputs)
                    
                    # Handle both standard transformers and ORTModelFeatureExtraction
                    # Yuan-embedding-2.0-en uses the CLS token (index 0)
                    if hasattr(outputs, "last_hidden_state"):
                        batch_vectors = outputs.last_hidden_state[:, 0, :]
                    else:
                        # Fallback for dictionary/tuple output
                        # In ORT, outputs[0] is typically the hidden states
                        batch_vectors = outputs[0][:, 0, :]
                        
                    batch_vectors = F.normalize(batch_vectors, p=2, dim=1)
                    
                    # Handle near-zero vectors by replacing them with a small noise vector 
                    # to satisfy Pinecone's non-zero requirement
                    sums = torch.abs(batch_vectors).sum(dim=1)
                    zero_mask = sums < 1e-6
                    if zero_mask.any():
                        print(f"Warning: Found {zero_mask.sum().item()} zero vectors. Adding epsilon.")
                        batch_vectors[zero_mask] = 1e-6

                    vectors.extend(batch_vectors.cpu().numpy())

            # 4. Batch Upsert to Pinecone

            if not skip_vector_upload:
                print(f"Upserting to Pinecone...")
                batch = []
                for chunk, vector in zip(chunks, vectors):
                    # Ensure ID is unique and fits within Pinecone's 512-character limit
                    # We truncate the section part just in case it's huge
                    safe_id = f"{chunk['doc_id']}-{chunk['section_id']}-{chunk['chunk_index']}"
                    if len(safe_id) > 500:
                        safe_id = safe_id[:500]

                    batch.append({
                        "id": safe_id,
                        "values": vector.tolist(),
                        "metadata": {
                            "text": chunk["content"],
                            "doc_id": chunk["doc_id"],
                            "section_id": chunk["section_id"],
                            "section_title": chunk["section_title"],
                            "chunk_index": chunk["chunk_index"],
                            "page_number": chunk["page_number"],
                            "citation": chunk["citation"],
                            "source_url": chunk["source_url"]
                        }
                    })

                    if len(batch) >= 50:
                        await asyncio.to_thread(index.upsert, vectors=batch)
                        batch = []

                if batch:
                    await asyncio.to_thread(index.upsert, vectors=batch)
            else:
                print(f"Skipping vector upload to Pinecone for Cap {cap_num}")

            processed_counter["count"] += 1
            print(f"--- Completed Cap {cap_num} ({len(chunks)} chunks) | Progress: {processed_counter['count']}/{processed_counter['total']} ---")

        except Exception as e:
            print(f"Error processing Cap {cap_num}: {e}")

async def ingest_legal_pdfs(cap_numbers=None, batch_size=3, embedding_batch_size=16, model=None, force_reprocess=False, skip_vector_upload=False):
    """
    Ingests specified Cap numbers or all available PDFs in data/pdfs.
    Processes in batches concurrently.
    """
    print(f"=== Starting Legal PDF Ingestion Pipeline (V2) - Concurrent Limit: {batch_size} | Embedding Batch: {embedding_batch_size} ===")
    
    # Robust path handling (Absolute Paths)
    pdf_dir = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data\pdfs"
    parsed_dir = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data\parsed"
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
    if not os.path.exists(parsed_dir):
        os.makedirs(parsed_dir, exist_ok=True)
    
    print(f"Target PDF directory: {pdf_dir}")
    print(f"Target Parsed directory: {parsed_dir}")
    
    # Initialize the Vector Store Manager to get Pinecone handle
    vsm = VectorStoreManager()
    index = vsm.pc.Index(vsm.index_name)
    
    # Initialize the Embedding Model once if not provided
    if model is None:
        print("Loading BOOSTED ONNX/TensorRT embedding model...")
        model_name = "IEITYuan/Yuan-embedding-2.0-en"
        model_path = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\yuan-onnx-trt"
        
        # Check if ONNX model exists, otherwise fallback to standard
        if os.path.exists(os.path.join(model_path, "model.onnx")):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": model_path
                },
                {}, # CUDA
                {}  # CPU
            ]
            
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_path,
                provider="TensorrtExecutionProvider",
                provider_options={
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": model_path
                }
            )
        else:
            print(f"Warning: ONNX model not found at {model_path}. Falling back to standard model.")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            model.eval()
    else:
        # If model is passed, we assume tokenizer is also needed
        tokenizer = AutoTokenizer.from_pretrained("IEITYuan/Yuan-embedding-2.0-en", trust_remote_code=True)
    
    processed_counter = {"count": 0, "total": 0}

    
    if not cap_numbers:
        # Get all cap numbers from filenames
        search_pattern = os.path.join(pdf_dir, "cap*.pdf")
        pdf_files = glob.glob(search_pattern)
        cap_numbers = [os.path.basename(f).replace("cap", "").replace(".pdf", "") for f in pdf_files]
        
        # Sort them numerically
        import re
        def cap_sort_key(c):
            match = re.match(r"(\d+)([A-Z]*)", c)
            if match:
                return (int(match.group(1)), match.group(2))
            return (0, c)
        cap_numbers.sort(key=cap_sort_key)

    processed_counter["total"] = len(cap_numbers)
    print(f"Total ordinances to process: {len(cap_numbers)}")

    # Implementation of concurrency logic
    semaphore = asyncio.Semaphore(batch_size)
    
    # Create all tasks
    tasks = [process_single_cap(cap, pdf_dir, parsed_dir, semaphore, index, model, tokenizer, processed_counter, embedding_batch_size, force_reprocess, skip_vector_upload) for cap in cap_numbers]

    
    # Run them!
    await asyncio.gather(*tasks)

    print(f"\n========================================")
    print(f"Ingestion Pipeline Complete!")
    print(f"Total PDFs successfully upserted: {processed_counter['count']}")
    print(f"========================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest legal PDFs into Pinecone.")
    parser.add_argument("--cap", nargs="+", help="Specific Cap numbers to process (e.g., --cap 282 1)")
    parser.add_argument("--batch", type=int, default=3, help="Number of files to process in parallel (default: 3)")
    parser.add_argument("--embedding_batch", type=int, default=16, help="Batch size for embedding model (default: 16)")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if JSON exists")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to Pinecone")
    args = parser.parse_args()
    
    asyncio.run(ingest_legal_pdfs(cap_numbers=args.cap, batch_size=args.batch, embedding_batch_size=args.embedding_batch, force_reprocess=args.force, skip_vector_upload=args.skip_upload))
