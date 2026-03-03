import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")


pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing. Add it to your .env file.")

index_name = os.getenv("PINECONE_INDEX", "my-vector-index")
#A check to see if index exists, if not it creates one
pc = Pinecone(api_key=pinecone_api_key)
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=len(embedding.embed_query("dimension check")),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )

index = pc.Index(index_name)

# Initialize PineconeVectorStore with the index and embedding function
vector_store = PineconeVectorStore(index=index, embedding=embedding)

results = vector_store.similarity_search(
    query="Who was the businessman and entrepreneur?", k=3
)

# Print the search results
print("Top match:")
for index, result in enumerate(results, start=1):
    print(f"{index}. {result.metadata.get('name', 'Unknown')} -> {result.page_content}")

# Function to fetch all document IDs
def get_all_document_ids():
    # Use a dummy query (e.g., an empty query or a very broad query) to retrieve all documents
    query = ""  # A neutral query or use a placeholder query
    top_k = 1000  # The number of documents to retrieve per query (adjust based on your index size)
    
    # Run the query to fetch the results
    results = vector_store.similarity_search(query=query, k=top_k)
    
    # Collect all document IDs
    document_ids = []
    for index, result in enumerate(results, start=1):
        document_ids.append(result.id)  # Get document ID

    return document_ids

# Fetch all document IDs
all_document_ids = get_all_document_ids()

# Print the document IDs
print("All document IDs:")
for doc_id in all_document_ids:
    print(doc_id)