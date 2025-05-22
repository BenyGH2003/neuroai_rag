import streamlit as st

st.set_page_config(page_title="NeuroAIHub: Neuroradiology Imaging Dataset Finder", layout="wide")

import pandas as pd
import re
import os
from tabulate import tabulate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

# --- API Key Setup ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.warning("⚠️ API key not found. Please set OPENAI_API_KEY in Streamlit secrets or environment.")
    st.stop()

# --- Load Data ---
@st.cache_resource
def load_data():
    sheets = ['Neurodegenerative', 'Neoplasm', 'Cerebrovascular', 'Psychiatric', 'Spinal', 'Neurodevelopmental']
    file_path = 'dataset.xlsx'
    dataframes = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
    return dataframes, sheets

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local("vectorstore_db", embeddings, allow_dangerous_deserialization=True)

dataframes, sheets = load_data()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# --- LLM Setup ---
class ChatOpenRouter(ChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=OPENAI_API_KEY,
            **kwargs
        )

llm = ChatOpenRouter(model_name="meta-llama/llama-3.3-8b-instruct:free")

template = """
You are a helpful assistant for querying a neuroradiology dataset...
Question: {question}
Context: {context}
Answer:
"""

PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

# --- Dataset Matching ---
def find_datasets(query):
    docs = retriever.get_relevant_documents(query)
    results = []
    seen = set()
    for doc in docs:
        content = doc.page_content
        for sheet, df in dataframes.items():
            for _, row in df.iterrows():
                row_tuple = tuple(row.values)
                if (str(row.get("dataset_name", "")) in content or str(row.get("disease", "")) in content) and (sheet, row_tuple) not in seen:
                    results.append((sheet, row))
                    seen.add((sheet, row_tuple))
    return results

def format_results(dataset_info):
    if not dataset_info:
        return "No relevant datasets found."
    display_data = []
    for sheet, row in dataset_info:
        row_data = {col: row.get(col, 'N/A') for col in row.index}
        row_data['category'] = sheet
        display_data.append(row_data)
    df = pd.DataFrame(display_data)
    cols = ['category'] + [col for col in df.columns if col != 'category']
    df = df[cols]
    return tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False)

# --- Streamlit UI ---
st.title("🧠 Explore a rich database of neuroradiology imaging datasets.")
st.markdown("Hello! I'm NeuroAIHub, your assistant for exploring neuroradiology datasets. Ask me anything!")

query = st.text_input("💬 Your Question:", placeholder="e.g., Show me datasets about Parkinson's disease")

if query:
    with st.spinner("🔍 Searching..."):
        answer = qa_chain.run(query)
        datasets = find_datasets(query)
        table = format_results(datasets)

    st.subheader("🧾 Answer:")
    st.write(answer)

    st.subheader("📊 Relevant Datasets:")
    st.code(table)
