import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)


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

vector_store = PineconeVectorStore(index=index, embedding=embedding)

retriever = MultiQueryRetriever.from_llm(retriever= vector_store.as_retriever(search_kwargs ={"k" : 3}), llm = model)
query = "Best universities in the world?"

results = retriever.invoke(query)

for i, result in enumerate(results):
    print(f"Result from Retriever {i+1}:")
    print(result.page_content)
    print("\n---\n")
############################################################################