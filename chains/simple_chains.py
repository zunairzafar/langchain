import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
prompt = PromptTemplate(

    template= "Generate detailed report on {topic}.",
    input_variables=["topic"]
)



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser #langchain expression to create a chain of operations

response = chain.invoke({'topic': ' cricket'})

print(response)
#chain.get_graph().print_ascii() to visualize the chain structure in ASCII format