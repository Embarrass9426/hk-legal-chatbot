import os
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "hk-legal-rag")
        
        if not self.api_key:
            print("WARNING: PINECONE_API_KEY not set.")
            return

        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384, # Dimension for all-MiniLM-L6-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.api_key
        )

    def upsert_documents(self, documents):
        """
        Upserts a list of document dictionaries into Pinecone.
        Each doc should have 'id', 'content', and 'metadata'.
        """
        if not self.api_key:
            return
            
        texts = [doc['content'] for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k != 'content'} for doc in documents]
        ids = [doc['id'] for doc in documents]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, k: int = 5):
        """
        Searches the vector store for relevant documents.
        """
        if not self.api_key:
            return []
            
        return self.vector_store.similarity_search(query, k=k)
