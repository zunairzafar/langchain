import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

st.header("AI Chatbot")
user_input = st.text_input("Enter your prompt")

llm = HuggingFaceEndpoint(

    repo_id="deepseek-ai/DeepSeek-V3.2",
    task = "text-generation"


)

model = ChatHuggingFace(llm=llm)


if st.button('Submit'):
    result = model.invoke(user_input)

    st.write(result.content)
