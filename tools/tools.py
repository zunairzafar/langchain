from langchain.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id= "openai/gpt-oss-20b",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
@tool
def multiply(a :int, b:int) -> int:
    """Multiplies two numbers."""
    return a * b

llm_with_tool = model.bind_tools([multiply])

query = HumanMessage(content="what is 23 times 2312")
messages = [query]

result = llm_with_tool.invoke(messages)
messages.append(result)

if result.tool_calls:
    tool_response = multiply.invoke(result.tool_calls[0])
    messages.append(tool_response)

final_response = llm_with_tool.invoke(messages)
print(final_response.content)

