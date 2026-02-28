import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") 

llm = HuggingFaceEndpoint(

    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"


)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    country: str = Field(description="The country of residence of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate name, age, country of a fictional {place} person in the format \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': "Pakistani"})

print(result)
