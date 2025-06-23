import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import requests
import pickle


load_dotenv()

# Function to get the TAIDE API token
def get_token():
    username = os.getenv("TAIDE_EMAIL")
    password = os.getenv("TAIDE_PASSWORD")
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token", data={"username": username, "password": password}
    )
    token = r.json()["access_token"]
    return token

# Initialize page settings with a wider layout
def init_page():
    st.set_page_config(page_title="Ask about PDF", layout="wide")
    st.session_state.costs = []
    st.session_state.feedback_log = []  # Initialize feedback log
    st.session_state.selected = None  # Keep track of the selected retrieval method

# Improved function to extract text from a PDF with detailed output
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
    
    st.info(f"Extracted text length: {len(text)} characters")
    
    # Print chunking settings
    st.info("Chunking text with chunk_size=1000 and chunk_overlap=200")
    
    # Increase chunk size and add overlap for better context retention
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",
        chunk_size=1000,  # Increased chunk size for more context
        chunk_overlap=200  # Added overlap to retain context
    )
    pdf_text = text_splitter.split_text(text)
    
    st.write(f"Total number of chunks created: {len(pdf_text)}")
    
    # Display the first 3 chunks with separation lines
    st.write("### First 3 Chunks:")
    for i, chunk in enumerate(pdf_text[:3]):
        st.write(f"Chunk {i+1}:\n")
        st.write(chunk)
        st.write("---")  # Separator between chunks
    
    return pdf_text

# Load all PDFs from the docs directory
def load_all_pdfs_from_directory(directory="./docs"):
    pdf_texts = []
    file_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            st.write(f"Loading PDF: {file_name}")
            pdf_text = extract_text_from_pdf(file_path)
            pdf_texts.append((file_name, pdf_text))  # Store with filename
            file_names.append(file_name)

    st.info(f"The files {', '.join(file_names)} are loaded successfully.")
    return pdf_texts, file_names

# Load Chroma DB for embeddings with additional logging
def load_chroma():
    st.info("Initializing HuggingFace embeddings with 'all-MiniLM-L6-v2'")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    st.info("Loading Chroma vector store from './data' directory")
    chroma = Chroma(
        embedding_function=embeddings,  
        persist_directory="./data"
    )
    
    st.success("Chroma DB loaded successfully!")
    return chroma

# Build vector store from extracted PDF text
def build_vector_store_from_all_pdfs(pdf_texts):
    chroma = load_chroma()
    
    st.info("Adding texts to the Chroma vector store and embedding them...")
    for file_name, pdf_text in pdf_texts:
        # Ensure metadata includes 'source' key for each chunk
        chroma.add_texts(pdf_text, metadatas=[{"source": file_name}] * len(pdf_text))  # Store metadata for source
    
    st.success("The vector store (DB) was constructed successfully.")
# Build the QA retrieval models for both MMR and Cosine Similarity
def build_qa_models():
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=200,
        timeout=None,
        max_retries=2,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )
    
    chroma = load_chroma()

    # MMR retriever
    st.info("Creating MMR retriever (Maximum Marginal Relevance) for diverse results.")
    retriever_mmr = chroma.as_retriever(
        search_type="mmr",  # Use MMR instead of pure similarity
        search_kwargs={"k": 5}
    )
    
    # Cosine Similarity retriever
    st.info("Creating Cosine Similarity retriever for relevance-based results.")
    retriever_cos = chroma.as_retriever(
        search_type="similarity",  # Use pure cosine similarity
        search_kwargs={"k": 5}
    )
    
    qa_mmr = RetrievalQA.from_chain_type(
        llm=taide_llm,
        chain_type="stuff", 
        retriever=retriever_mmr,
        return_source_documents=True,
        verbose=True
    )
    
    qa_cos = RetrievalQA.from_chain_type(
        llm=taide_llm,
        chain_type="stuff", 
        retriever=retriever_cos,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_mmr, qa_cos

# Function to display documents and separate them with lines, showing the source
def display_documents(documents, title):
    st.write(f"## {title}")
    displayed_documents = set()
    for doc in documents:
        # Check if 'source' key exists in metadata
        source = doc.metadata.get('source', 'Unknown Source')  # Default to 'Unknown Source' if missing
        if doc.page_content not in displayed_documents:
            st.write(f"Source: {source}")
            st.write(doc.page_content)
            st.write("---")  # Separator between documents
            displayed_documents.add(doc.page_content)

# Function to handle the query and return results
def ask(qa, query):
    st.info("Sending query to the QA model...")
    answer = qa.invoke(query)
    return answer

# Function to save feedback locally in a JSON file
# Function to save feedback locally in a JSON file with logging and error handling
def save_feedback(query, answer, feedback, document_contents):
    try:
        # Log entry to be saved
        log_entry = {
            "query": query,
            "answer": answer,
            "feedback": feedback,
            "retriever_method": st.session_state.selected,  # Selected retrieval method
            "document_contents": [doc.page_content for doc in document_contents]  # Save text from documents
        }

        # Append to session state
        st.session_state.feedback_log.append(log_entry)

        # Path to the feedback log file
        file_path = "./feedback_log.json"

        # Check if the file already exists
        if os.path.exists(file_path):
            # Open the existing file and load its content
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            # Start a new list if the file doesn't exist
            data = []

        # Add the new log entry to the data
        data.append(log_entry)

        # Save the updated data back to the JSON file
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        # Display success message
        st.success("Your feedback and document contents have been saved successfully!")

        # Optionally display the saved feedback for debugging
        st.json(log_entry)  # Show the feedback log in JSON format on the screen

    except Exception as e:
        st.error(f"An error occurred while saving feedback: {e}")
        st.write(f"Error details: {e}")


# Main function for a single-page app
def main():
    init_page()

    st.title("Ask about PDF")

    pdf_texts, file_names = load_all_pdfs_from_directory()

    if pdf_texts:
        st.success(f"{len(file_names)} PDFs loaded from the 'docs' directory.")
        
        build_vector_store_from_all_pdfs(pdf_texts)

        query = st.text_input("Query: ", key="input")

        if query:
            qa_mmr, qa_cos = build_qa_models()
            
            if qa_mmr and qa_cos:
                with st.spinner("Typing answer..."):
                    # Get results from both MMR and Cosine Similarity
                    answer_mmr = ask(qa_mmr, query)
                    answer_cos = ask(qa_cos, query)

                # Display results side by side
                col1, col2 = st.columns(2)

                with col1:
                    display_documents(answer_mmr["source_documents"], "MMR Retrieval Results")

                with col2:
                    display_documents(answer_cos["source_documents"], "Cosine Similarity Retrieval Results")
                
                # Let the user choose the preferred result
                # ボタンの色を保持するための修正
                st.markdown("### Which result was better?")
                col1, col2 = st.columns([1, 1])

                # ユーザーが選択した結果をセッションステートで保持する
                if "selected" not in st.session_state:
                    st.session_state.selected = None

                # ボタンのスタイルを変えるための色付けロジック
                mmr_button_color = "secondary"  # 未選択時はデフォルトの色
                cos_button_color = "secondary"

                if st.session_state.selected == "MMR":
                    mmr_button_color = "danger"  # 選択時の色
                elif st.session_state.selected == "Cosine":
                    cos_button_color = "danger"  # 選択時の色

                with col1:
                    if st.button("MMR is better", key="mmr_button", use_container_width=True):
                        st.session_state.selected = "MMR"
                with col2:
                    if st.button("Cosine Similarity is better", key="cos_button", use_container_width=True):
                        st.session_state.selected = "Cosine"

                # 選択された結果に基づいて最終的な回答を生成
                if st.session_state.selected:
                    if st.session_state.selected == "MMR":
                        final_answer = answer_mmr
                    else:
                        final_answer = answer_cos
                    
                    st.markdown(f"## Final Answer (from {st.session_state.selected})")
                    st.write(final_answer["result"])

                    # Add feedback system
                    st.markdown("### How was the answer?")
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        good_button = st.button("👍 Good", key="good_button")
                    with col2:
                        not_good_button = st.button("👎 Not Good", key="not_good_button")

                    if good_button:
                        save_feedback(query, final_answer["result"], "Good", final_answer["source_documents"])
                    elif not_good_button:
                        save_feedback(query, final_answer["result"], "Not Good", final_answer["source_documents"])

                    st.write("Feedback Log:", st.session_state.feedback_log)

    else:
        st.warning("No PDFs found in the 'docs' directory.")

if __name__ == '__main__':
    main()
