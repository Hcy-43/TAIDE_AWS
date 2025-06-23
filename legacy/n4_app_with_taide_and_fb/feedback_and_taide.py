# Import necessary libraries
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA  # Deprecated 
import json
import requests

load_dotenv()

# Function to get the TAIDE API token
def get_token():
    username = os.getenv("TAIDE_EMAIL")
    password = os.getenv("TAIDE_PASSWORD")
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token", data={"username": username, "password": password}
    )
    token = r.json()["access_token"]
    print(token)
    return token

# Initialize TAIDE LLM with token
taide_llm = ChatOpenAI(
    model="TAIDE/a.2.0.0",
    temperature=0,
    max_tokens=200,
    timeout=None,
    max_retries=2,
    openai_api_base="https://td.nchc.org.tw/api/v1/",
    openai_api_key=get_token(),
)


# Initialize page settings
def init_page():
    st.set_page_config(
        page_title="Ask about PDF",
    )
    st.session_state.costs = []
    st.session_state.feedback_log = []  # Initialize feedback log


# Function to upload and extract text from PDF
def get_pdf_text_and_filename():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here',
        type='pdf'
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            chunk_size=500,
            chunk_overlap=0,
        )
        pdf_text = text_splitter.split_text(text)
        return pdf_text, uploaded_file.name  # return file text and file name
    else:
        return None, None


# Load Chroma DB for embeddings
def load_chroma():
    return Chroma(
        embedding_function=HuggingFaceEmbeddings(),
        persist_directory="./data"
    )


# Build vector store from extracted PDF text
def build_vector_store(pdf_text):
    chroma = load_chroma()
    chroma.add_texts(pdf_text)


# Build the QA retrieval model
def build_qa_model_with_filename(llm, pdf_text, file_name):
    chroma = load_chroma()
    chroma.add_texts(pdf_text)

    retriever = chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    ), file_name


# Function to save feedback locally in a JSON file
def save_feedback(query, answer, feedback):
    # Create the dictionary to log
    log_entry = {
        "query": query,
        "answer": answer,
        "feedback": feedback
    }

    # Append to session log
    st.session_state.feedback_log.append(log_entry)

    # Save to a local JSON file
    file_path = "./feedback_log.json"
    if os.path.exists(file_path):
        # If file exists, load existing data
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new entry
    data.append(log_entry)

    # Write back to file
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    st.success("Your feedback has been saved!")


# Function to handle the query and return results
def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)

    return answer, cb.total_cost


# Main function for a single-page app
def main():
    init_page()

    st.title("Ask about PDF")

    # Upload the PDF and display success message if uploaded
    pdf_text, file_name = get_pdf_text_and_filename()
    
    if pdf_text and file_name:
        st.success(f"File '{file_name}' uploaded successfully.")
        # Build vector store after PDF is uploaded
        build_vector_store(pdf_text)

        # Display query input field
        query = st.text_input("Query: ", key="input")

        # Ensure a question is entered
        if query:
            # Build the QA model and process the query
            qa, file_name = build_qa_model_with_filename(taide_llm, pdf_text, file_name)
            
            if qa:
                with st.spinner("Typing answer..."):
                    answer, cost = ask(qa, query)

                if answer:
                    st.markdown("## Answer")
                    st.write(answer["result"])
                    st.markdown("## Source Documents")
                    displayed_documents = set()
                    for doc in answer["source_documents"]:
                        if doc.page_content not in displayed_documents:
                            st.write(f"File Name: {file_name}")  # Display file name
                            st.write(doc.page_content)
                            displayed_documents.add(doc.page_content)

                    # Feedback buttons with icons
                    st.markdown("### How was the answer?")
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        good_button = st.button("👍 Good", key="good_button")
                    with col2:
                        not_good_button = st.button("👎 Not Good", key="not_good_button")

                    # Record feedback in session_state and save locally
                    if good_button:
                        save_feedback(query, answer["result"], "Good")
                    elif not_good_button:
                        save_feedback(query, answer["result"], "Not Good")

                    # Display feedback log for debugging or future use
                    st.write("Feedback Log:", st.session_state.feedback_log)
    else:
        st.warning("Please upload a PDF file to proceed.")


if __name__ == '__main__':
    main()
