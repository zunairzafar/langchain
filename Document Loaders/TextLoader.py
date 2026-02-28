#Uses text file and and converts them into document objects.
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
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


loader = TextLoader("text.txt", encoding="utf-8")
documents = loader.load()

content = documents[0].page_content

chain = prompt | model | parser

response = chain.invoke({"text": content})
print(response)