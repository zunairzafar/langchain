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

    template= "Write a detailed note on the {topic}",
    input_variables=['topic']

)
#template 2

template2 = PromptTemplate(
    template = "Summarize the {text} in 2 lines",
    input_variables=['text']

)

prompt1 = template1.format(topic="Earth's gravity")
result = model.invoke(prompt1)
prompt2 = template2.format(text=result.content)

result2 = model.invoke(prompt2)
print(result2.content)