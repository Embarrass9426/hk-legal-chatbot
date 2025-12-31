import asyncio
from pdf_parser import PDFLegalParser
from vector_store import VectorStoreManager
import os

async def ingest_cap282():
    print("Starting ingestion for Cap 282 (Employees' Compensation Ordinance)...")
    
    # 1. Parse PDF
    parser = PDFLegalParser("282")
    if not await parser.download_pdf():
        print("Failed to download PDF. Aborting.")
        return
        
    sections = parser.parse_sections()
    print(f"Successfully parsed {len(sections)} sections.")
    
    # 2. Initialize Vector Store
    vsm = VectorStoreManager()
    
    # 3. Upsert to Pinecone
    print("Upserting to Pinecone...")
    vsm.upsert_documents(sections)
    print("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(ingest_cap282())
