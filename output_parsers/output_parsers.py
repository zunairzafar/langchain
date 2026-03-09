import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(

    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"


)

model = ChatHuggingFace(llm=llm)

#template 1 

template1 = PromptTemplate(

    template= "Write a note on the {topic}",
    input_variables=['topic']

)
#template 2

template2 = PromptTemplate(
    template = "summarize the folowing text in 1 line: {text}",
    input_variables=['text']

)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Proxmia Centauri"})

print(result)
from langchain.tools import tool

