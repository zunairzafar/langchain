import os
from typing import TypedDict, Optional, Literal
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(

    repo_id="openai/gpt-oss-120b",
    task = "text-generation"


)

class Review(TypedDict):
    summary : str
    sentiment : str

model = ChatHuggingFace(llm=llm)

struct = model.with_structured_output(Review)

response = model.invoke("""The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
compared to other brands. Hoping for a software update to fix this.""")

print(response.content)