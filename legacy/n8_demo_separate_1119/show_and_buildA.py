import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv(".env")

# Function to extract text and metadata from a PDF using Azure DI
def extract_text_from_pdf_with_azure_di(pdf_path):
    try:
        # Azure DI credentials from environment variables

        # doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        doc_intelligence_endpoint = "https://llmchat.cognitiveservices.azure.com/"

        # doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        doc_intelligence_key = "1lENInNTwn3cZPGMXZOwYy3FSs0PFHxuYnEBg41PV0a192MhUDAkJQQJ99ALACYeBjFXJ3w3AAALACOG7Mt9"

        print(doc_intelligence_endpoint)
        
        # Initialize Azure Document Intelligence loader
        loader = AzureAIDocumentIntelligenceLoader(
            file_path=pdf_path,
            api_key=doc_intelligence_key,
            api_endpoint=doc_intelligence_endpoint,
            api_model="prebuilt-layout"
        )
        
        # Load and process the PDF document
        docs = loader.load()
        if not docs:
            st.error(f"No content extracted from {pdf_path}.")
            return []
        
        text_content = docs[0].page_content  # Extract the document content as text

        # Chunk the text using a Markdown header splitter
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks = text_splitter.split_text(text_content)

        if not chunks:
            st.warning(f"No chunks were generated from {pdf_path}.")
        return chunks
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {str(e)}")
        return []

# Function to initialize and persist chunks to a Chroma database
def process_and_store_in_chroma_db(pdf_files, db_dir="azure_db_temp"):
    try:
        # Initialize Chroma DB with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chroma = Chroma(embedding_function=embeddings, persist_directory=db_dir)
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join("docs", "doc_pdf", pdf_file)
            st.write(f"Processing and storing: {pdf_file}")
            
            # Extract text and chunks
            chunks = extract_text_from_pdf_with_azure_di(pdf_path)
            if chunks:
                for chunk in chunks:
                    # Add chunks to Chroma DB
                    chroma.add_texts(
                        [chunk.page_content],
                        metadatas=[{**chunk.metadata, "source": pdf_file}]
                    )
                
                # Display extracted chunks
                st.success(f"Successfully processed and stored chunks from {pdf_file}")
                st.write("### Extracted Chunks:")
                for i, chunk in enumerate(chunks):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(f"**Metadata:** {chunk.metadata}")
                    st.write(f"**Content:**\n{chunk.page_content}")
                    st.write("---")
            else:
                st.warning(f"No chunks were generated for {pdf_file}.")
        
        st.success(f"All chunks have been stored in the Chroma DB at '{db_dir}'.")
    except Exception as e:
        st.error(f"Error processing or storing chunks: {str(e)}")

# Streamlit interface
def main():
    st.title("PDF Text and Metadata Viewer with Azure DI")
    
    # Define the directory containing the PDF files
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "doc_pdf")
    
    # Get list of PDFs in the specified directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        st.warning("No PDF files found in the 'docs/doc_pdf' directory.")
        return

    st.write("### PDF Files in 'docs/doc_pdf' Directory:")
    for pdf_file in pdf_files:
        st.write(f"- {pdf_file}")
    
    # Button to process and store all PDFs
    if st.button("Process and Store All PDFs in Chroma DB"):
        process_and_store_in_chroma_db(pdf_files)

if __name__ == "__main__":
    main()
