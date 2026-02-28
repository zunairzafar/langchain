import os
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Function to load chat history from a file or create a new one
def load_chat_history(filename='D:\\lagchain_tutorial\\chatbot\\chat.json'):
    # Check if the file exists, if not, return a default SystemMessage
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            chat_history_dict = json.load(f)
        # Convert the dictionary back into message objects
        chat_history = []
        for message in chat_history_dict:
            if message["role"] == "SystemMessage":
                chat_history.append(SystemMessage(content=message["content"]))
            elif message["role"] == "HumanMessage":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "AIMessage":
                chat_history.append(AIMessage(content=message["content"]))
        return chat_history
    else:
        # If file doesn't exist, start a new chat with a default SystemMessage
        return [SystemMessage(content='You are a teacher who is expert in physics. If you do not know something, simply reply you have limited information on the topic')]

# Function to save chat history to a file
def save_chat_history(chat_history, filename='D:\\lagchain_tutorial\\chatbot\\chat.json'):
    # Convert chat history to a dictionary
    chat_history_dict = []
    for message in chat_history:
        message_dict = {
            "role": message.__class__.__name__,
            "content": message.content
        }
        chat_history_dict.append(message_dict)

    # Save to the file
    with open(filename, 'w') as f:
        json.dump(chat_history_dict, f, indent=4)