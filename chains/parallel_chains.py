import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
load_dotenv()  # Load environment variables from .env file
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

#Qwen/Qwen3-Coder-Next
llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task = 'text-generation'
)
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task = 'text-generation'
)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template= "Generate short and simple notes from the following text \n {text}.",
    input_variables=["text"]
)   

prompt2 = PromptTemplate(
    template= "Generate 5 short questions quiz from the following text -> \n {text}",
    input_variables=["text"]
)   

prompt3 = PromptTemplate(
    template= "Merge the notes and quiz  \n -> {notes} and {quiz}",  
    input_variables=["notes", "quiz"]
)


parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
#chain.get_graph().print_ascii() # to visualize the chain structure in ASCII format


text = """
Neural machine translation is a newly emerging approach to machine translation, recently proposed
by Kalchbrenner and Blunsom (2013), Sutskever et al. (2014) and Cho et al. (2014b). Unlike the
traditional phrase-based translation system (see, e.g., Koehn et al., 2003) which consists of many
small sub-components that are tuned separately, neural machine translation attempts to build and
train a single, large neural network that reads a sentence and outputs a correct translation.
Most of the proposed neural machine translation models belong to a family of encoder–
decoders (Sutskever et al., 2014; Cho et al., 2014a), with an encoder and a decoder for each language,
or involve a language-specific encoder applied to each sentence whose outputs are then compared
(Hermann and Blunsom, 2014). An encoder neural network reads and encodes a source sentence
into a fixed-length vector. A decoder then outputs a translation from the encoded vector. The
whole encoder–decoder system, which consists of the encoder and the decoder for a language pair,
is jointly trained to maximize the probability of a correct translation given a source sentence.
A potential issue with this encoder–decoder approach is that a neural network needs to be able to
compress all the necessary information of a source sentence into a fixed-length vector. This may
make it difficult for the neural network to cope with long sentences, especially those that are longer
than the sentences in the training corpus. Cho et al. (2014b) showed that indeed the performance of
a basic encoder–decoder deteriorates rapidly as the length of an input sentence increases.
In order to address this issue, we introduce an extension to the encoder–decoder model which learns
to align and translate jointly. Each time the proposed model generates a word in a translation, it
(soft-)searches for a set of positions in a source sentence where the most relevant information is
concentrated. The model then predicts a target word based on the context vectors associated with
these source positions and all the previous generated target words.

"""
# Now run it
response = chain.invoke({'text': text})
print(response)