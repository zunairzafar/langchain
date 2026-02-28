from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

loader = WebBaseLoader("https://www.bbc.com/news/articles/cdjmrxkwk3mo")
documents = loader.load()
content = documents[0].page_content

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = PromptTemplate(
    template= "Summarize the following text -> \n {text}.",
    input_variables=["text"]
)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)   
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
chain = prompt | model | parser
response = chain.invoke({"text": content})
print(response)