import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.documents import Document
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

llm_pipeline = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_length=512
)

API_URLS = {
    'arxiv': 'http://export.arxiv.org/api/query',
    'semanticscholar': 'https://api.semanticscholar.org/graph/v1/paper/search'
}

def fetch_papers(query, source, max_results=5):
    params = {'search_query': query, 'start': 0, 'max_results': max_results} if source == 'arxiv' else {'query': query, 'limit': max_results}
    response = requests.get(API_URLS[source], params=params)
    response.raise_for_status()
    return response.json() if source == 'semanticscholar' else response.text

def create_documents(data, source):
    documents = []
    if source == 'arxiv':
        for entry in data.split('<entry>')[1:]:
            title = entry.split('<title>')[1].split('</title>')[0].strip()
            summary = entry.split('<summary>')[1].split('</summary>')[0].strip()
            documents.append(Document(page_content=summary, metadata={'title': title}))
    else:
        for paper in data.get('data', []):
            title = paper.get('title', 'No title available')
            abstract = paper.get('abstract', '')
            documents.append(Document(page_content=abstract, metadata={'title': title}))
    return documents

st.set_page_config(page_title="AI-Powered Academic Assistant", layout="wide")
st.markdown("""
    <style>
        body, .main, .stApp {
            background: #0E1117;
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            background: #1E1E1E !important;
            color: #FFFFFF !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 8px !important;
            padding: 12px !important;
        }
        .stTextInput label, .stTextArea label, .stSelectbox label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        .stButton>button {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: transform 0.2s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        .stMarkdown h1 {
            color: #FFFFFF !important;
            border-bottom: 2px solid #6366F1;
            padding-bottom: 0.5rem;
        }
        .response-container {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 2rem;
            margin-top: 1.5rem;
            border: 1px solid #4A4A4A;
        }
        .stSpinner > div > div {
            border-color: #6366F1 transparent transparent transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI-Powered Academic Assistant")
st.markdown("### Your Smart Research Companion")
source = st.selectbox("Select data source:", ['arxiv', 'semanticscholar'])
topic = st.text_input("Enter the academic topic:")
query = st.text_area("Enter your academic query (e.g., 'list top papers', 'summarize topic', 'find most cited articles'):")

if st.button("🚀 Generate Insights"):
    with st.spinner("🔍 Analyzing research papers..."):
        api_response = fetch_papers(topic, source)
        documents = create_documents(api_response, source)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])[:512]
        response = llm_pipeline.invoke(f"{query}\nContext: {context}")
        st.subheader("AI Response:")
        st.write(response)
