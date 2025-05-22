import streamlit as st
import warnings
from dotenv import load_dotenv
import pandas as pd
import re
import os
from tabulate import tabulate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

warnings.filterwarnings("ignore", category=UserWarning)

# --- API Key Setup ---
load_dotenv()
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
You are a helpful assistant for querying a neuroradiology dataset. 
You have access to datasets from six categories:
Neurodegenerative, Neoplasm, Cerebrovascular, Psychiatric, Spinal, Neurodevelopmental.

Based on the following question and the retrieved context:
- Identify which datasets are relevant
- Summarize the dataset names, diseases, and modalities

Question: {question}
Context: {context}
Answer:
"""

PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- Functions from original code ---

def find_relevant_datasets(query):
    """Find datasets relevant to the query"""
    # Use the retriever to get relevant documents
    relevant_docs = retriever.get_relevant_documents(query)

    # Extract dataset information from relevant documents
    dataset_info = []
    # Use a set to keep track of processed rows to avoid duplicates
    processed_rows_set = set()

    # First try to find exact matches from the documents
    for doc in relevant_docs:
        content = doc.page_content
        for sheet_name, df in dataframes.items():
            for index, row in df.iterrows():
                # Check if any dataset name or disease from this row is in the document
                dataset_name = str(row.get('dataset_name', ''))
                disease = str(row.get('disease', ''))
                # Convert the row to a tuple of its values for reliable comparison
                row_tuple = tuple(row.values)

                if (dataset_name in content or disease in content) and (sheet_name, row_tuple) not in processed_rows_set:
                    # Append the original row object along with sheet name
                    dataset_info.append((sheet_name, row))
                    # Add the tuple representation to the set
                    processed_rows_set.add((sheet_name, row_tuple))


    # If no matches found, extract possible disease/condition keywords and search again
    if not dataset_info:
        # Common neurological conditions to look for
        conditions = ['alzheimer', 'parkinson', 'stroke', 'tumor', 'cancer', 'schizophrenia',
                     'bipolar', 'autism', 'epilepsy', 'dementia', 'multiple sclerosis',
                     'brain', 'spine', 'neural', 'cerebral', 'neurodegenerative']

        condition_pattern = '|'.join(conditions)
        matches = re.findall(condition_pattern, query.lower())

        if matches:
            for match in matches:
                for sheet_name, df in dataframes.items():
                    for index, row in df.iterrows():
                        disease = str(row.get('disease', '')).lower()
                        # Convert the row to a tuple of its values
                        row_tuple = tuple(row.values)

                        if match in disease and (sheet_name, row_tuple) not in processed_rows_set:
                            # Append the original row object along with sheet name
                            dataset_info.append((sheet_name, row))
                            # Add the tuple representation to the set
                            processed_rows_set.add((sheet_name, row_tuple))

    # If still no matches, return datasets from categories mentioned in the query
    if not dataset_info:
        for sheet_name in sheets:
            if sheet_name.lower() in query.lower():
                for index, row in dataframes[sheet_name].iterrows():
                     # Convert the row to a tuple of its values
                    row_tuple = tuple(row.values)
                    if (sheet_name, row_tuple) not in processed_rows_set:
                         # Append the original row object along with sheet name
                        dataset_info.append((sheet_name, row))
                         # Add the tuple representation to the set
                        processed_rows_set.add((sheet_name, row_tuple))


    # If still no matches, return a sample from all categories
    if not dataset_info:
        for sheet_name, df in dataframes.items():
            if not df.empty:
                # Convert the row to a tuple of its values
                row = df.iloc[0]
                row_tuple = tuple(row.values)
                if (sheet_name, row_tuple) not in processed_rows_set:
                     # Append the original row object along with sheet name
                    dataset_info.append((sheet_name, row))
                     # Add the tuple representation to the set
                    processed_rows_set.add((sheet_name, row_tuple))


    return dataset_info

def format_table(dataset_info, query):
    """Format dataset information as a beautiful table with all columns"""
    if not dataset_info:
        return "No relevant datasets found."

    # Filter datasets based on disease keywords if present in the query
    disease_pattern = re.compile(r'(alzheimer|parkinson|stroke|tumor|cancer|schizophrenia|bipolar|autism|epilepsy|dementia)', re.IGNORECASE)
    disease_matches = disease_pattern.findall(query.lower())

    filtered_info = []
    if disease_matches:
        for sheet, row in dataset_info:
            disease = str(row.get('disease', '')).lower()
            if any(d in disease for d in disease_matches):
                filtered_info.append((sheet, row))
    else:
        filtered_info = dataset_info

    # If filtering removed all results, revert to original list
    if not filtered_info:
        filtered_info = dataset_info

    # Create a DataFrame with all columns
    display_data = []
    for sheet, row in filtered_info:
        row_data = {col: row.get(col, 'N/A') for col in row.index}
        row_data['category'] = sheet  # Add the sheet name as a category
        display_data.append(row_data)

    display_df = pd.DataFrame(display_data)

    # Move category to first column
    if 'category' in display_df.columns:
        cols = display_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('category')))
        display_df = display_df[cols]

    # Use tabulate for a more beautiful table
    table = tabulate(display_df, headers='keys', tablefmt='fancy_grid', showindex=False)
    return table

def is_category_query(query):
    """Check if the query is asking about a general category of datasets"""
    # Look for category names in the query
    query_lower = query.lower()
    
    # Look for patterns like "show me spinal datasets" or "neoplasm datasets"
    category_patterns = [
        f"{category.lower()} datasets" for category in sheets
    ]
    
    # Add more query patterns
    category_patterns.extend([
        f"{category.lower()} data" for category in sheets
    ])
    
    # Also check for phrases like "datasets in the spinal category"
    category_patterns.extend([
        f"datasets in the {category.lower()} category" for category in sheets
    ])
    
    # Check for more general category queries
    for pattern in category_patterns:
        if pattern in query_lower:
            # Extract the category name
            for category in sheets:
                if category.lower() in pattern:
                    return category
    
    # Check for direct category mentions
    for category in sheets:
        if category.lower() == query_lower or f"about {category.lower()}" in query_lower:
            return category
    
    return None

def get_category_overview(category):
    """Generate an overview of datasets in a specific category"""
    # Get the dataframe for this category
    df = dataframes[category]
    
    # Count number of datasets
    dataset_count = len(df)
    
    # Get list of diseases covered
    diseases = df['disease'].dropna().unique()
    disease_list = ", ".join(str(d) for d in diseases if str(d) != 'nan')
    
    # Get common modalities
    modalities = df['modality'].dropna().unique()
    modality_list = ", ".join(str(m) for m in modalities if str(m) != 'nan')
    
    # Calculate average number of subjects
    try:
        avg_subjects = df['subject_no_f'].dropna().astype(float).mean()
        avg_subjects_str = f"{avg_subjects:.1f}"
    except:
        avg_subjects_str = "varies"
    
    # Years range
    try:
        years = df['year'].dropna().astype(int)
        year_range = f"{years.min()}-{years.max()}" if not years.empty else "various years"
    except:
        year_range = "various years"
    
    # Create overview text
    overview = f"""
The {category} category contains {dataset_count} datasets focusing on {category.lower()} conditions and related neurological aspects.

Disease focus: {disease_list}

These datasets primarily use {modality_list} imaging modalities, with an average of {avg_subjects_str} subjects per dataset, collected during {year_range}.

Below is the complete list of datasets in this category.
"""
    return overview.strip()

def process_query(query):
    """Process the user query and return both a text response and dataset table"""
    # Check if this is a general category query
    category = is_category_query(query)
    
    if category:
        # Generate a general overview of the category
        text_response = get_category_overview(category)
        
        # Get all datasets in this category for the table
        mentioned_datasets = []
        df = dataframes[category]
        for index, row in df.iterrows():
            mentioned_datasets.append((category, row))
        
        # Format the table with all datasets in the category
        table_response = format_table(mentioned_datasets, query)
        
        return text_response, table_response
    
    # For non-category queries, proceed with the existing logic
    # Always get text response from RAG
    text_response = qa_chain.run(query)
    
    # Find all datasets mentioned in the text response
    mentioned_datasets = []
    
    # Get all datasets from all sheets
    all_datasets = []
    for sheet_name, df in dataframes.items():
        for index, row in df.iterrows():
            all_datasets.append((sheet_name, row))
    
    # Extract dataset names and diseases mentioned in the text response
    for sheet_name, row in all_datasets:
        dataset_name = str(row.get('dataset_name', ''))
        disease = str(row.get('disease', ''))
        
        # Check if this dataset name is explicitly mentioned in the text response
        if dataset_name and dataset_name in text_response:
            mentioned_datasets.append((sheet_name, row))
        # Also check if the disease is mentioned along with identifying context
        elif disease and disease in text_response:
            # Look for more specific mentions to avoid false positives
            context_terms = [sheet_name, str(row.get('modality', '')), str(row.get('institution', ''))]
            for term in context_terms:
                if term and term in text_response:
                    mentioned_datasets.append((sheet_name, row))
                    break
    
    # If no datasets were explicitly mentioned, fall back to the regular search
    if not mentioned_datasets:
        # Find relevant datasets based on the query
        dataset_info = find_relevant_datasets(query)
        # Extract dataset names to look for in the response
        for sheet_name, row in dataset_info:
            dataset_name = str(row.get('dataset_name', ''))
            if dataset_name and dataset_name in text_response:
                mentioned_datasets.append((sheet_name, row))
    
    # If still no datasets found, just show a limited selection from query results
    if not mentioned_datasets:
        dataset_info = find_relevant_datasets(query)
        # Limit to max 5 datasets to avoid overwhelming the user
        mentioned_datasets = dataset_info[:min(5, len(dataset_info))]
    
    # Format the table with only the mentioned datasets
    table_response = format_table(mentioned_datasets, query)
    
    return text_response, table_response

# --- Streamlit UI ---

st.set_page_config(page_title="NeuroAIHub: Neuroradiology Imaging Dataset Finder", layout="wide")
st.title("🧠 Explore a rich database of neuroradiology imaging datasets.")
st.markdown("Hello! I'm NeuroAIHub, your assistant for exploring neuroradiology datasets. Ask me anything!")

query = st.text_input("💬 Your Question:", placeholder="e.g., Show me datasets about Parkinson's disease")

if query:
    with st.spinner("🔍 Searching..."):
        text_response, table_response = process_query(query)

    st.subheader("🧾 Answer:")
    st.write(text_response)

    st.subheader("📊 Relevant Datasets:")
    st.code(table_response)
