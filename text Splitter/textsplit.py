from langchain_text_splitters import  CharacterTextSplitter

text = "This is a sample text that we will use to demonstrate how to split text into smaller chunks using the CharacterTextSplitter from the langchain_text_splitters library. The CharacterTextSplitter allows us to specify a chunk size and an overlap size, which can be useful for processing large texts in smaller, more manageable pieces." \
" It is important to choose an appropriate chunk size and overlap size based on the specific use case and the nature of the text being processed. By splitting the text into smaller chunks, we can improve the efficiency of various natural language processing tasks, such as summarization, question answering, and more."


text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=3, separator='')
#use 10-20% chunk overlap to ensure that the context is preserved across chunks, especially for tasks that require understanding of the text as a whole, such as summarization or question answering. The separator parameter can be set to an empty string to split the text based on character count without adding any additional characters between chunks.
chunks = text_splitter.split_text(text)
print(chunks)