import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone_class import PineconeDocumentStore

store = PineconeDocumentStore()

# Example documents to add
#documents = [
   ## {"content": "Elon Musk is a businessman and entrepreneur. He is the richest man in the world", "metadata": {"name": "Elon Musk", "field":"businessman"}},
   # {"content": "Jeff Bezos is the founder of Amazon. He owns private jets, and islands.", "metadata": {"name": "Jeff Bezos", "field":"businessman"}},
   # {"content": "Bill Gates co-founded Microsoft. He aims to build computer use for everyone", "metadata": {"name": "Bill Gates", "field":"businessman"}},
#]

#store.add_documents(documents)

query = "Who was the pilot?"
results = store.vector_store.similarity_search(query=query, k=1)

print("Top match:")
for index, result in enumerate(results, start=1):
    print(f"{index}. {result.metadata.get('name', 'Unknown')} -> {result.page_content}")
