import os
import torch
import torch.nn.functional as F
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv
from vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    # 1. Initialize VectorStoreManager (uses BoostedYuanEmbeddings with CLS pooling)
    print("--- Initializing VectorStoreManager ---")
    vsm = VectorStoreManager()
    index = vsm.pc.Index(vsm.index_name)

    # 2. Define the chunk data
    chunk_data = {
        "content": "Interpretation and General Clauses Ordinance(Cap. 1)ContentsSectionPagePart IShort Title and Application1.Short title1-22.Application1-22A.Laws previously in force1-4Part IIInterpretation of Words and Expressions3.Interpretation of words and expressions2-23AA.References to upholding Basic Law and bearing allegiance to HKSAR2-363A.References to former or retired judge2-424.(Repealed)2-445.Grammatical variations and cognate expressions2-446.References to Government property2-447.Provisions for gender and number2-448.Service by post2-469.Chinese and English words and expressions2-46Last updated date18.12.2025",
        "page_number": 1,
        "chunk_index": 1,
        "section_id": "preamble",
        "section_title": "Preamble",
        "doc_id": "cap1",
        "citation": "Cap. 1, Preamble"
    }

    # 3. Test ID Formats
    id_format_ingest = f"cap1_chunk_1"
    id_format_manager = f"cap1-preamble-1"
    
    print(f"Checking for IDs: {id_format_ingest}, {id_format_manager}")
    
    fetch_response = index.fetch(ids=[id_format_ingest, id_format_manager])
    
    found_id = None
    stored_vector = None
    
    for vid in [id_format_ingest, id_format_manager]:
        if vid in fetch_response.vectors:
            print(f"Found vector with ID: {vid}")
            found_id = vid
            stored_vector = fetch_response.vectors[vid].values
            break
            
    if not found_id:
        print("[ERROR] Could not find either ID in Pinecone. Here are some vectors in the index:")
        # Try to find anything cap1 related
        results = index.query(vector=[0]*1024, top_k=5, filter={"doc_id": "cap1"}, include_metadata=True)
        if results.matches:
            print(f"Found {len(results.matches)} matches for 'cap1' filter. Example ID: {results.matches[0].id}")
            found_id = results.matches[0].id
            stored_vector = index.fetch(ids=[found_id]).vectors[found_id].values
        else:
            print("No vectors found for cap1.")
            return

    # 4. Generate local embedding using the current model in VectorStoreManager
    # Note: VectorStoreManager.upsert_chunks uses a prefix. 
    # ingest_legal_pdfs.py uses '{title}\n{content}'.
    
    # We will try both to see which matches better.
    
    text_variant_1 = f"Represent this legal document passage for retrieval: {chunk_data['content']}"
    text_variant_2 = f"{chunk_data['section_title']}\n{chunk_data['content']}"
    
    print("\n--- Generating Local Embeddings ---")
    
    # Using the embedding class from VectorStoreManager
    emb_v1 = vsm.embeddings.embed_query(text_variant_1)
    emb_v2 = vsm.embeddings.embed_query(text_variant_2)

    print("Stored vector norm:", np.linalg.norm(stored_vector))
    print("Local emb_v1 norm:", np.linalg.norm(emb_v1))
    print("Local emb_v2 norm:", np.linalg.norm(emb_v2))
    
    # 5. Calculate Similarity
    sim1 = cosine_similarity(emb_v1, stored_vector)
    sim2 = cosine_similarity(emb_v2, stored_vector)
    
    print(f"\nComparing with stored ID: {found_id}")
    print(f"Similarity (Prefix: 'Represent this...'): {sim1:.4f}")
    print(f"Similarity (Prefix: 'Preamble\\n...'): {sim2:.4f}")
    
    if max(sim1, sim2) > 0.99:
        print("\nMATCH FOUND! The local model is consistent with the stored embeddings.")
    elif max(sim1, sim2) > 0.9:
        print("\nCLOSE MATCH. The model is likely the same, but there might be slight pooling or precision differences.")
    else:
        print("\n[ERROR] NO MATCH. The model or processing logic (pooling/normalization) differs significantly.")
        print("Distance check: The stored vector and local vector are different.")

if __name__ == "__main__":
    main()
