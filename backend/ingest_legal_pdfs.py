import asyncio
import os
import glob
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*max_size parameter is deprecated.*")

from pdf_parser_v2 import PDFLegalParserV2
from vector_store import VectorStoreManager
from sentence_transformers import SentenceTransformer

async def process_single_cap(cap_num, pdf_dir, semaphore, index, model, processed_counter, embedding_batch_size=16, layout_batch_size=8, force_reprocess=False):
    """Processes a single ordinance cap with batch embedding and enriched metadata."""
    async with semaphore:
        try:
            parser = PDFLegalParserV2(cap_num, pdf_dir=pdf_dir)
            json_path = os.path.join(parser.output_dir, f"cap{cap_num}.json")
            
            # Skip if JSON already exists (parsing and embedding)
            if not force_reprocess and os.path.exists(json_path):
                processed_counter["count"] += 1
                print(f"--- Skipping Cap {cap_num} (JSON found) | Progress: {processed_counter['count']}/{processed_counter['total']} ---")
                return

            print(f"\n--- Starting Cap {cap_num} ---")

            # 1. Parse PDF (Heavy CPU/GPU task)
            chunks = await asyncio.to_thread(parser.process_ordinance, layout_batch_size=layout_batch_size)
            if not chunks:
                return

            # 2. Clear old embeddings from Pinecone
            await asyncio.to_thread(index.delete, filter={"doc_id": f"cap{cap_num}"})

            # 3. Batch Embed semantic text (section title + content)
            print(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [
                f"{c['section_title']}\n{c['content']}"
                for c in chunks
            ]

            vectors = await asyncio.to_thread(
                model.encode,
                texts,
                batch_size=embedding_batch_size,
                show_progress_bar=False
            )

            # 4. Batch Upsert to Pinecone
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

            processed_counter["count"] += 1
            print(f"--- Completed Cap {cap_num} ({len(chunks)} chunks) | Progress: {processed_counter['count']}/{processed_counter['total']} ---")

        except Exception as e:
            print(f"Error processing Cap {cap_num}: {e}")

async def ingest_legal_pdfs(cap_numbers=None, batch_size=3, embedding_batch_size=16, model=None, layout_batch_size=8, force_reprocess=False):
    """
    Ingests specified Cap numbers or all available PDFs in data/pdfs.
    Processes in batches concurrently.
    """
    print(f"=== Starting Legal PDF Ingestion Pipeline (V2) - Concurrent Limit: {batch_size} | Embedding Batch: {embedding_batch_size} | Layout Batch: {layout_batch_size} ===")
    
    # Robust path handling
    if os.path.exists("backend/data/pdfs"):
        pdf_dir = "backend/data/pdfs"
    elif os.path.exists("data/pdfs"):
        pdf_dir = "data/pdfs"
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(script_dir, "data", "pdfs")
    
    print(f"Target PDF directory: {os.path.abspath(pdf_dir)}")
    
    # Initialize the Vector Store Manager to get Pinecone handle
    vsm = VectorStoreManager()
    index = vsm.pc.Index(vsm.index_name)
    
    # Initialize the Embedding Model (SentenceTransformer) once if not provided
    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer("IEITYuan/Yuan-embedding-2.0-en", trust_remote_code=True)
    
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
    tasks = [process_single_cap(cap, pdf_dir, semaphore, index, model, processed_counter, embedding_batch_size, layout_batch_size, force_reprocess) for cap in cap_numbers]
    
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
    parser.add_argument("--layout_batch", type=int, default=8, help="Batch size for layout model (default: 8)")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if JSON exists")
    args = parser.parse_args()
    
    asyncio.run(ingest_legal_pdfs(cap_numbers=args.cap, batch_size=args.batch, embedding_batch_size=args.embedding_batch, layout_batch_size=args.layout_batch, force_reprocess=args.force))
