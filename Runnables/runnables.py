import os
import langchain
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # Load environment variables from .env file
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

