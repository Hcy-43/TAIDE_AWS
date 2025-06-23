import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import requests
import pickle
import time
from langchain.schema import HumanMessage, AIMessage
import openai

load_dotenv()

# Function to initialize the page
def init_page():
    # Set up the Streamlit page configuration (e.g., title, layout)
    st.set_page_config(page_title="Ask about PDF", layout="wide")
    
    # Initialize session state variables
    if "costs" not in st.session_state:
        st.session_state.costs = []
        
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []

# Function to get the TAIDE API token
def get_token():
    username = os.getenv("TAIDE_EMAIL")
    password = os.getenv("TAIDE_PASSWORD")
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token", data={"username": username, "password": password}
    )
    token = r.json()["access_token"]
    return token

# Retry logic for handling rate limit errors
def retry_with_backoff(func, max_retries=5, delay=10):
    retries = 0
    while retries < max_retries:
        try:
            return func()  # Try to call the function
        except openai.error.RateLimitError:
            retries += 1
            st.warning(f"Rate limit reached. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            break
    return None  # Return None if all retries fail

# Function to perform semantic chunking using a model like TAIDE
def perform_semantic_chunking(text, taide_llm):
    initial_chunks = text.split('\n\n')  # Simple paragraph-based splitting
    semantic_chunks = []
    current_chunk = ""

    for paragraph in initial_chunks:
        current_chunk += paragraph + "\n\n"
        query = f"Is the following text a complete and semantically coherent unit? Text: {current_chunk}"
        messages = [HumanMessage(content=query)]

        # Call the TAIDE LLM to check if the chunk is semantically meaningful
        def call_model():
            return taide_llm(messages)

        response = retry_with_backoff(call_model)  # Retry on rate limit

        if response and isinstance(response, AIMessage) and "Yes" in response.content:  # If yes, accept the chunk as complete
            semantic_chunks.append(current_chunk.strip())
            current_chunk = ""  # Reset for the next chunk

        time.sleep(2)  # Wait for 2 seconds before making another request

    if current_chunk:
        semantic_chunks.append(current_chunk.strip())

    return semantic_chunks

# Function to extract text from a PDF and apply semantic chunking
def extract_text_from_pdf_with_semantic_chunking(pdf_path, taide_llm):
    pdf_reader = PdfReader(pdf_path)
    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
    chunks = perform_semantic_chunking(text, taide_llm)
    return chunks

# Load all PDFs from the docs directory
def load_all_pdfs_from_directory(taide_llm, directory="./docs"):
    pdf_texts = []
    file_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            pdf_text = extract_text_from_pdf_with_semantic_chunking(file_path, taide_llm)
            pdf_texts.append(pdf_text)
            file_names.append(file_name)

    st.info(f"The files {', '.join(file_names)} are loaded successfully.")
    return pdf_texts, file_names

# Load Chroma DB for embeddings
def load_chroma():
    return Chroma(
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory="./data"
    )

# Build vector store from extracted PDF text
def build_vector_store_from_all_pdfs(pdf_texts):
    chroma = load_chroma()
    for pdf_text in pdf_texts:
        chroma.add_texts(pdf_text)
    st.success("The database (DB) was constructed successfully.")

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

    # Using Maximum Marginal Relevance (MMR) for more diverse retrieval results
    retriever = chroma.as_retriever(
        search_type="mmr",  
        search_kwargs={"k": 3}
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

    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=100,
        timeout=None,
        max_retries=2,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )

    pdf_texts, file_names = load_all_pdfs_from_directory(taide_llm)

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
