import os
import json
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from chat_utils import load_chat_history, save_chat_history

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

llm = HuggingFaceEndpoint(

    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm =llm)
chat_history = load_chat_history()  # This loads chat history from chat.json


while True:
    user_input = input("You...: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break

    response = model.invoke(chat_history)
    print("A.I : ", response.content)
    chat_history.append(AIMessage(content=response.content))


# Save chat history to a JSON file
save_chat_history(chat_history=chat_history)
