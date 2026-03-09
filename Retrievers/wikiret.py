from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results= 3, lang="en")

query = "What happened to khamnei?"

docs  = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("\n---\n")
