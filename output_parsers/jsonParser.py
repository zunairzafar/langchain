import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(

    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"


)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
#template 1 

template = PromptTemplate(

    template= "My name is Zunair. I am from Pakistan and doing M.S in Computer Engineering. \n{format_instruction}",

    input_variables=[],

    partial_variables={'format_instruction': parser.get_format_instructions()}

)
chain = template | model | parser 

result = chain.invoke({})

print(result)

#In case of JsonOutputParser, the LLM decides itself the format of the output, there is no dfaukt way. 






