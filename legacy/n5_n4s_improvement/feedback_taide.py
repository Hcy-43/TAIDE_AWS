import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA  # Deprecated 
import json
import requests
import pickle
from pydantic.v1 import root_validator, validator

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

# Initialize page settings
def init_page():
    st.set_page_config(page_title="Ask about PDF")
    st.session_state.costs = []
    st.session_state.feedback_log = []  # Initialize feedback log

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",
        chunk_size=500,
        chunk_overlap=0,
    )
    pdf_text = text_splitter.split_text(text)
    return pdf_text

# Load all PDFs from the docs directory
def load_all_pdfs_from_directory(directory="./docs"):
    pdf_texts = []
    file_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            pdf_text = extract_text_from_pdf(file_path)
            pdf_texts.append(pdf_text)
            file_names.append(file_name)

    # Display loaded files for debugging
    st.info(f"The files {', '.join(file_names)} are loaded successfully.")
    print(f"The files {', '.join(file_names)} are loaded successfully (printed for debugging).")

    return pdf_texts, file_names

# Load Chroma DB for embeddings (without caching)
def load_chroma():
    return Chroma(
        embedding_function=HuggingFaceEmbeddings(),
        persist_directory="./data"
    )

# Build vector store from extracted PDF text
def build_vector_store_from_all_pdfs(pdf_texts):
    chroma = load_chroma()
    for pdf_text in pdf_texts:
        chroma.add_texts(pdf_text)
    st.success("The database (DB) was constructed successfully.")
    print("The database (DB) was constructed successfully (printed for debugging).")

# Build the QA retrieval model (LLM is created each time)
def build_qa_model():
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

    retriever = chroma.as_retriever(
        search_type="similarity", # Can be “similarity” (default), “mmr”, or “similarity_score_threshold”.
        search_kwargs={"k": 10}
    )
    return RetrievalQA.from_chain_type(
        llm=taide_llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

# Function to save feedback locally in a JSON file
def save_feedback(query, answer, feedback, document_contents):
    log_entry = {
        "query": query,
        "answer": answer,
        "feedback": feedback,
        "document_contents": document_contents
    }

    st.session_state.feedback_log.append(log_entry)

    file_path = "./feedback_log.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    data.append(log_entry)

    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    st.success("Your feedback and document contents have been saved!")

# Function to handle the query and return results
def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa.invoke(query)
    return answer, cb.total_cost

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
            qa = build_qa_model()
            
            if qa:
                with st.spinner("Typing answer..."):
                    answer, cost = ask(qa, query)

                if answer:
                    st.markdown("## Answer")
                    st.write(answer["result"])
                    st.markdown("## Source Documents")
                    displayed_documents = set()
                    document_contents = []
                    for doc in answer["source_documents"]:
                        if doc.page_content not in displayed_documents:
                            st.write(f"File Name: {', '.join(file_names)}")
                            st.write(doc.page_content)
                            displayed_documents.add(doc.page_content)
                            document_contents.append(doc.page_content)

                    st.markdown("### How was the answer?")
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        good_button = st.button("👍 Good", key="good_button")
                    with col2:
                        not_good_button = st.button("👎 Not Good", key="not_good_button")

                    if good_button:
                        save_feedback(query, answer["result"], "Good", document_contents)
                    elif not_good_button:
                        save_feedback(query, answer["result"], "Not Good", document_contents)

                    st.write("Feedback Log:", st.session_state.feedback_log)
    else:
        st.warning("No PDFs found in the 'docs' directory.")


if __name__ == '__main__':
    main()
