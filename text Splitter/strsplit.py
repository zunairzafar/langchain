from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "This is a sample text that we will use to demonstrate how to split text into smaller chunks using the CharacterTextSplitter from the langchain_text_splitters library. The CharacterTextSplitter allows us to specify a chunk size and an overlap size, which can be useful for processing large texts in smaller, more manageable pieces." \
" It is important to choose an appropriate chunk size and overlap size based on the specific use case and the nature of the text being processed. By splitting the text into smaller chunks, we can improve the efficiency of various natural language processing tasks, such as summarization, question answering, and more."

text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=3)

chunks = text_splitter.split_text(text)
print(chunks)