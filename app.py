import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# 1. Load environment variables
api_key = st.secrets["HUGGINGFACE_API_KEY"]
# 2. Initialize the Hugging Face Inference Client
client = InferenceClient(model="facebook/bart-large-cnn", token=api_key)

def summarize_hf_api(text, max_len=100, min_len=30):
    """
    Summarize text using the Hugging Face InferenceClient (serverless).
    Pass max_length and min_length via the 'parameters' dictionary.
    """
    summary_text = client.summarization(
        text,
        parameters={
            "max_length": max_len,
            "min_length": min_len,
            "do_sample": False
        }
    )
    return summary_text.summary_text
# Streamlit app
st.title("PDF Summarizer (Hugging Face InferenceClient)")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
if pdf_file is not None:
    # Write uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()

    # Split the PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Summarize each chunk
    summaries = []
    for doc in docs:
        chunk_text = doc.page_content
        summary = summarize_hf_api(chunk_text, max_len=100, min_len=30)
        summaries.append(summary)

    # Combine partial summaries
    combined_text = " ".join(summaries)
    final_summary = summarize_hf_api(combined_text, max_len=150, min_len=50)

    st.write("### Final Summary:")
    st.write(final_summary)
