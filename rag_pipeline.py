import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_core.documents import Document
import streamlit as st

def fetch_arxiv_papers(query, max_results=5):
    base_url = 'http://export.arxiv.org/api/query'
    params = {'search_query': query, 'start': 0, 'max_results': max_results}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch from arXiv API. Status code: {response.status_code}")

def create_documents_from_metadata(api_response):
    documents = []
    for entry in api_response.split('<entry>')[1:]:
        title = entry.split('<title>')[1].split('</title>')[0].strip()
        summary = entry.split('<summary>')[1].split('</summary>')[0].strip()
        documents.append(Document(page_content=summary, metadata={'title': title}))
    return documents

def main():
    st.title("AI-Powered Academic Assistant")
    st.write("Search and get AI-generated academic answers from arXiv papers.")

    topic = st.text_input("Enter the academic topic to search on arXiv:")
    query = st.text_input("Enter your academic query:")

    if st.button("Submit"):
        api_response = fetch_arxiv_papers(topic)
        documents = create_documents_from_metadata(api_response)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])[:512]
        llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
        response = llm_pipeline(f"{query}\nContext: {context}", max_new_tokens=100)[0]['generated_text']
        st.write("## AI Response:")
        st.write(response)

if __name__ == "__main__":
    main()

