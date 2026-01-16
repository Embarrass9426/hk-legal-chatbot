import asyncio
import time
import sys
import os

# Add current directory to path so we can import ingest_legal_pdfs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest_legal_pdfs import ingest_legal_pdfs

async def run_optimization():
    # Use specified caps for testing
    test_caps = ["6B", "6", "6A", "4", "3", "A501", "A601", "A7", "A602", "A305"]
    concurrency_range = range(3, 11)  # 3 to 10
    embedding_batches = [22]
    layout_batches = [4, 8, 16]
    
    from sentence_transformers import SentenceTransformer
    print("Pre-loading embedding model for benchmark...")
    model = SentenceTransformer("IEITYuan/Yuan-embedding-2.0-en", trust_remote_code=True)
    
    results = []
    
    print(f"Starting optimization with {len(test_caps)} caps...")
    print(f"Concurrency range: {list(concurrency_range)}")
    print(f"Embedding batch sizes: {embedding_batches}")
    print(f"Layout batch sizes: {layout_batches}")
    
    for cb in concurrency_range:
        for eb in embedding_batches:
            for lb in layout_batches:
                print(f"\n{'='*20}")
                print(f"TESTING: Concurrency={cb}, Embedding Batch={eb}, Layout Batch={lb}")
                print(f"{'='*20}")
                
                start_time = time.time()
                try:
                    # Run the ingestion pipeline without uploading to Pinecone to speed up benchmarking
                    await ingest_legal_pdfs(cap_numbers=test_caps, batch_size=cb, embedding_batch_size=eb, model=model, layout_batch_size=lb, force_reprocess=True, skip_vector_upload=True)
                    duration = time.time() - start_time
                    
                    print(f"\n[RESULT] Time for CB={cb}, EB={eb}, LB={lb}: {duration:.2f} seconds")
                    results.append(((cb, eb, lb), duration))
                except Exception as e:
                    print(f"\n[ERROR] Failed combination CB={cb}, EB={eb}, LB={lb}: {e}")

    # Summary table
    print("\n" + "="*80)
    print(f"{'Concurrency':<12} | {'Embed Batch':<12} | {'Layout Batch':<12} | {'Time (s)':<10}")
    print("-" * 65)
    for (cb, eb, lb), duration in results:
        print(f"{cb:<12} | {eb:<12} | {lb:<12} | {duration:<10.2f}")
    print("="*80)
    
    if results:
        best_cfg, best_time = min(results, key=lambda x: x[1])
        print("\n" + "*"*60)
        print(f"BEST CONFIGURATION FOUND:")
        print(f"Shortest Time: {best_time:.2f}s")
        print(f"Concurrency: {best_cfg[0]}")
        print(f"Embedding Batch: {best_cfg[1]}")
        print(f"Layout Batch: {best_cfg[2]}")
        print("*"*60)
    else:
        print("\nNo successful runs were completed.")

if __name__ == "__main__":
    asyncio.run(run_optimization())
