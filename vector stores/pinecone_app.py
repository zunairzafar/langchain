import streamlit as st
from pinecone_class import PineconeDocumentStore

# Initialize the PineconeDocumentStore class
store = PineconeDocumentStore()

def add_documents():
    """Function to handle adding documents."""
    st.subheader("Add Documents")
    num_docs = st.number_input("How many documents do you want to add?", min_value=1, step=1)

    documents = []
    custom_ids = []
    for i in range(num_docs):
        content = st.text_area(f"Document Content {i+1}", key=f"content_{i}")
        metadata_str = st.text_input(f"Metadata {i+1} -> Only accepts string value", key=f"metadata_{i}")  # Accept metadata as string
        custom_id = st.text_input(f"Custom ID for Document {i+1}", key=f"custom_id_{i}")
        
        # Convert metadata string to a dictionary (you can adjust the structure as needed)
        metadata = {"class": metadata_str}  # Adjust the metadata dictionary based on your needs
        
        documents.append({"content": content, "metadata": metadata})
        custom_ids.append(custom_id)

    if st.button("Add Documents"):
        if documents:
            store.add_documents(documents, custom_ids)
            st.success(f"{num_docs} documents added successfully!")


def remove_documents():
    """Function to handle removing documents."""
    st.subheader("Remove Documents")
    doc_ids = st.text_area("Enter Document IDs (comma-separated)", key="remove_ids").split(",")
    doc_ids = [doc_id.strip() for doc_id in doc_ids]

    if st.button("Remove Documents"):
        if doc_ids:
            store.remove_documents(doc_ids)
            st.success(f"{len(doc_ids)} documents removed successfully!")

def update_documents():
    """Function to handle updating documents."""
    st.subheader("Update Documents")
    doc_ids = st.text_area("Enter Document IDs (comma-separated) to Update", key="update_ids").split(",")
    doc_ids = [doc_id.strip() for doc_id in doc_ids]

    new_docs = []
    custom_ids = []
    for i in range(len(doc_ids)):
        content = st.text_area(f"Updated Content for Document {i+1}", key=f"update_content_{i}")
        metadata = st.text_input(f"Updated Metadata {i+1}", key=f"update_metadata_{i}")
        new_docs.append({"content": content, "metadata": metadata})
        custom_id = st.text_input(f"Custom ID for Document {i+1}", key=f"update_custom_id_{i}")
        custom_ids.append(custom_id)

    if st.button("Update Documents"):
        if doc_ids and new_docs:
            store.update_documents(doc_ids, new_docs)
            st.success(f"{len(doc_ids)} documents updated successfully!")
def query_documents():
    """Function to handle querying documents."""
    st.subheader("Query Documents")
    query = st.text_input("Enter your query", key="query_input")
    top_k = st.number_input("Top K Document Results", min_value=1, step=1, value=10)

    if st.button("Query Documents"):
        if query:
            results = store.query_documents(query, top_k)
            st.write("Query Results:")
            for result in results:
                st.write(f"ID: {result['id']}, Metadata: {result['metadata']}, Content: {result['content']}")


def get_all_document_ids():
    """Function to retrieve and display all document IDs."""
    st.subheader("Get All Document IDs")

    if st.button("Get All Document IDs"):
        document_ids = store.get_all_document_ids()
        if document_ids:
            st.write("Document IDs:")
            st.write(document_ids)
            st.success(f"Retrieved {len(document_ids)} document ID(s).")
        else:
            st.info("No document IDs found in the index.")



# Streamlit Layout
st.title("Pinecone Document Management")

query_documents()

get_all_document_ids()

# Add document functionality
add_documents()

# Remove document functionality
remove_documents()

# Update document functionality
update_documents()