import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

st.header("Dynamic Prompting")

llm = HuggingFaceEndpoint(

    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"


)

model = ChatHuggingFace(llm=llm)




prompt_template = load_prompt('prompt_generator.json')


if st.button("Summarize"):
    formatted_prompt = prompt_template.format(paper_input=paper_input, style_input=style_input, length_input=length_input)
    response = model.invoke(formatted_prompt)
    st.write(response.content)