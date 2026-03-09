from langchain.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import json
load_dotenv()

import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id= "openai/gpt-oss-20b",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Gets the conversion factor between two currencies."""
    url = f"https://v6.exchangerate-api.com/v6/7e41e328cfbc77bb1e02ec3f/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    return response.json()


@tool
def convert(base_currency: int, conversion_rate : Annotated[float, InjectedToolArg]) -> float:
    """Converts the base currency to the target currency using the conversion rate."""
    result = base_currency * conversion_rate
    formatted_result = f"{result:.2f}"
    return formatted_result

llm_with_tools = model.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage(content="what is the latest conversion rate between USD and EUR, based on the information convert 24 usd to eur?")]
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)
    if tool_call['name'] == "convert":
        tool_message2 = convert.invoke(tool_call, conversion_rate=conversion_rate)
        messages.append(tool_message2)

print(llm_with_tools.invoke(messages).content)