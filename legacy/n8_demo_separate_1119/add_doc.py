import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
import warnings

# Suppress the FutureWarning from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Load environment variables
load_dotenv()

# Function to load and process CSV content
def load_csv_content(csv_path):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_path)
        if data.empty:
            st.error(f"No content found in {csv_path}.")
            return ""

        # Combine rows into a single text block if necessary
        combined_content = "\n".join(data.astype(str).apply(lambda x: " | ".join(x), axis=1))
        return combined_content
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return ""

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=700, chunk_overlap=200):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

# Function to store chunks in Chroma DB
def store_chunks_in_chroma_db(chunks, metadata, db_dir="azure_db"):
    try:
        # Initialize Chroma DB with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chroma = Chroma(embedding_function=embeddings, persist_directory=db_dir)

        # Add chunks to Chroma DB
        chroma.add_texts(
            texts=chunks,
            metadatas=[metadata] * len(chunks)  # Ensure metadata is repeated for all chunks
        )
        st.success("Chunks successfully added to Chroma DB.")
    except Exception as e:
        st.error(f"Error storing chunks in Chroma DB: {str(e)}")

# Main function
def main():
    st.title("Add CSV Content to Chroma DB")

    # CSV file path
    csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.csv")

    # Check if CSV exists
    if not os.path.exists(csv_file_path):
        st.error(f"CSV file not found at {csv_file_path}. Please ensure the file exists.")
        return

    st.write(f"Processing file: {csv_file_path}")

    # Load CSV content
    csv_content = load_csv_content(csv_file_path)
    if not csv_content:
        return

    # Split content into chunks
    chunks = split_text_into_chunks(csv_content)
    if not chunks:
        st.warning("No chunks generated from the CSV content.")
        return

    # Define metadata for the CSV content
    metadata = {
        "source": os.path.basename(csv_file_path),
        "type": "CSV",
        "description": "Content from output.csv"
    }

    # Store chunks in Chroma DB
    store_chunks_in_chroma_db(chunks, metadata)

    # Display the processed chunks
    st.write("### Processed Chunks:")
    for i, chunk in enumerate(chunks):
        st.write(f"**Chunk {i + 1}:** {chunk}")
        st.write("---")

if __name__ == "__main__":
    main()
