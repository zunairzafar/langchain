import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

class PineconeDocumentStore:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

        # Initialize the embedding function
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Initialize Pinecone
        self.pinecone_api_key = st.secrets["general"]["streamlit_api_key"]
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is missing. Add it to your .env file.")
        
        self.index_name = os.getenv("PINECONE_INDEX", "my-vector-index")

        # Set up Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self._check_or_create_index()

        # Initialize the vector store
        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding)

    def _check_or_create_index(self):
        """Check if the Pinecone index exists, if not, create it."""
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            print(f"Index {self.index_name} does not exist. Creating it...")
            self.pc.create_index(
                name=self.index_name,
                dimension=len(self.embedding.embed_query("dimension check")),
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )

    def add_documents(self, docs: list, custom_ids: list = None):
        """Add documents to the Pinecone index with optional custom IDs."""
        if custom_ids and len(custom_ids) != len(docs):
            raise ValueError("Number of custom IDs must match the number of documents.")

        documents = []
        for i, doc in enumerate(docs):
            doc_id = custom_ids[i] if custom_ids else None  # Use custom ID if provided, else let Pinecone handle it
            documents.append(Document(page_content=doc['content'], metadata=doc['metadata'], id=doc_id))
        
        self.vector_store.add_documents(documents)
        print(f"{len(docs)} documents added successfully.")

    def remove_documents(self, doc_ids: list):
        """Remove documents from the Pinecone index by their custom IDs."""
        self.vector_store.delete(ids=doc_ids)
        print(f"{len(doc_ids)} documents removed successfully.")

    def update_documents(self, doc_ids: list, new_docs: list):
        """Update documents in the Pinecone index by their custom IDs."""
        updated_docs = [Document(page_content=doc['content'], metadata=doc['metadata'], id=doc_ids[i]) for i, doc in enumerate(new_docs)]
        self.vector_store.update(ids=doc_ids, documents=updated_docs)
        print(f"{len(doc_ids)} documents updated successfully.")

    def get_all_document_ids(self, namespace: str = ""):
        """Retrieve all document IDs from the Pinecone index."""
        all_ids = []

        if hasattr(self.index, "list"):
            try:
                for batch in self.index.list(namespace=namespace):
                    if isinstance(batch, list):
                        all_ids.extend(batch)
                    elif isinstance(batch, dict):
                        if "ids" in batch and isinstance(batch["ids"], list):
                            all_ids.extend(batch["ids"])
                        elif "vectors" in batch and isinstance(batch["vectors"], list):
                            all_ids.extend([item.get("id") for item in batch["vectors"] if item.get("id")])
            except TypeError:
                for batch in self.index.list():
                    if isinstance(batch, list):
                        all_ids.extend(batch)

        if not all_ids and hasattr(self.index, "list_paginated"):
            pagination_token = None
            while True:
                response = self.index.list_paginated(namespace=namespace, pagination_token=pagination_token, limit=100)

                vectors = response.get("vectors", []) if isinstance(response, dict) else []
                all_ids.extend([item.get("id") for item in vectors if item.get("id")])

                pagination = response.get("pagination", {}) if isinstance(response, dict) else {}
                pagination_token = pagination.get("next")
                if not pagination_token:
                    break

        return all_ids
    
    def query_documents(self, query: str, top_k=3):
                # Perform the similarity search
        results = self.vector_store.similarity_search(query=query, k=top_k)

        # Use enumeration and display the query results
        documents = []
        for index, result in enumerate(results, start=1):
            documents.append({
                'id': result.id,
                'metadata': result.metadata,
                'content': result.page_content
            })

        return documents
