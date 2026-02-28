import os
import warnings
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

documents = ["Sachin Tendulkar is Known as the Master Blaster, he holds numerous records, including the most runs in both Tests and ODI",
             "Virat Kohli is A modern-day legend, known for his consistency and aggressive batting across all formats.",
             "Shane Warne is  One of the greatest leg spinners in history, with over 700 Test wickets.",
             "Jacques Kallis is A brilliant all-rounder from South Africa, excelling in both batting and bowling.",
             "MS Dhoni is Renowned for his leadership, he led India to victory in three major ICC tournaments."]


query = "Who has the most test wickets?"

#in this code, every time the user query is independent, means no memory. It is done with the vector databases.

document_embedding = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

result =  (cosine_similarity([query_embedding], document_embedding))[0]

index = np.argmax(np.array(result))
max_score = np.max(np.array(result))
print(f"All scores are {result} " , "\nMax score at", documents[index], "\nMax score is ", max_score  )

