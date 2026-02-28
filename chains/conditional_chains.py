import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()  # Load environment variables from .env file
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

llm1= HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = 'text-generation'
)
model1 = ChatHuggingFace(llm=llm1)


#we need a structured output , as we need positive, negative or neutral as output. So we will use PydanticOutputParser to parse the output of the model into a structured format.
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="The sentiment of the text")
   # reason: str = Field(description="A brief explanation of why this sentiment was chosen")


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        "You are a strict JSON generator.\n"
        "Return ONLY a valid JSON object (no prose, no code fences, no markdown).\n"
        "The JSON MUST match this schema:\n"
        "{format_instructions}\n\n"
        "Text: {text}\n"
    ),
    input_variables = ["text"],
    partial_variables= {'format_instructions': parser2.get_format_instructions()}
)
classifier_chain = prompt1 | model1 | parser2


pos_prompt = PromptTemplate(
    template = "Generate a response to the positive feedback -> \n {feedback}",
    input_variables = ["feedback"]
)


neg_prompt = PromptTemplate(
    template = "Generate a response to the negative feedback -> \n {feedback}",
    input_variables = ["feedback"]
)
# We keep BOTH the original text and the label by using assign()
chain = (
    RunnablePassthrough.assign(label=classifier_chain)
    | RunnableBranch(
        (lambda x: x["label"].sentiment == "positive", pos_prompt | model1 | StrOutputParser()),
        (lambda x: x["label"].sentiment == "negative", neg_prompt | model1 | StrOutputParser()),
        RunnableLambda(lambda x: "Thank you for your feedback!"),
    )
)

text = "very very good product."
response = chain.invoke({"text": text, "feedback": text})
print(response)