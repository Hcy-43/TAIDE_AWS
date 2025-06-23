import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import requests
import pickle

load_dotenv()

st.set_page_config(page_title="Ask about Markdown", layout="wide")

# Function to get the TAIDE API token
def get_token():
    username = os.getenv("TAIDE_EMAIL")
    password = "taidetaide"
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token", data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if r.status_code == 200:
        try:
            token = r.json()["access_token"]
            if token:
                return token
            else:
                raise ValueError("Access token not found in the response.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from the server.")
    else:
        raise ConnectionError(f"Failed to retrieve token. Status code: {r.status_code}, Response: {r.text}")

# Initialize page settings with a wider layout
# def init_page():
#     st.set_page_config(page_title="Ask about Markdown", layout="wide")
#     st.session_state.costs = []
#     st.session_state.feedback_log = []  # Initialize feedback log
#     st.session_state.selected = None  # Keep track of the selected retrieval method

# Function to load and chunk Markdown content from files
def load_all_md_from_directory(directory="../docs/三大癌症_md"):
    md_texts = []
    file_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".md"):
            file_path = os.path.join(directory, file_name)
            st.write(f"Loading Markdown file: {file_name}")
            with open(file_path, "r", encoding="utf-8") as file:
                md_content = file.read()
            
            # Set up the Markdown header splitter to chunk based on headers
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                # ("###", "Header 3"),
            ]
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            chunks = text_splitter.split_text(md_content)
            
            st.write(f"Total number of chunks created for {file_name}: {len(chunks)}")
            st.write("### First 3 Chunks:")
            for i, chunk in enumerate(chunks[:3]):
                st.write(f"--- Chunk {i+1} ---")
                st.write(f"Metadata: {chunk.metadata}")
                st.write(f"Content:\n{chunk.page_content}")
                st.write("---")  # Separator between chunks
            
            md_texts.append((file_name, chunks))  # Store chunks with filename
            file_names.append(file_name)

    st.info(f"The files {', '.join(file_names)} are loaded successfully.")
    return md_texts, file_names

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

# Build vector store from extracted Markdown text
def build_vector_store_from_all_markdown(md_texts):
    chroma = load_chroma()
    
    st.info("Adding texts to the Chroma vector store and embedding them...")
    for file_name, md_chunks in md_texts:
        # Extract the page content (as string) from each chunk
        text_content = [chunk.page_content for chunk in md_chunks]  # Ensure you are working with text
        
        # Store metadata for source
        chroma.add_texts(text_content, metadatas=[{"source": file_name}] * len(text_content))
    
    st.success("The vector store (DB) was constructed successfully.")


from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Define the system and human message templates
system_template = """
您是一位獻身且專業的助手，專門協助回答患者的問題。請遵循以下指引：

- 僅根據文件內容進行回答，並提供準確且易於理解的資訊。
- 若文件中找不到相關資訊，請告知「目前的資訊無法回答您的問題」。
- 請勿自行猜測答案，若無確切資訊，請誠實地說明。
- 以親切且讓患者感到安心的語氣回答問題。
"""

human_template = """
您是一位醫療支援的專業助手，請根據以下文件內容，用親切且易懂的語氣回答患者的問題：
{context}

如果無法找到相關答案，請明確告知「目前的資訊無法回答您的問題」。
"""

# Combine the system and human message templates into a chat prompt
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])


# Load Chroma DB for embeddings
def load_chroma():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma = Chroma(
        embedding_function=embeddings,  
        persist_directory="./data"
    )
    return chroma

# Function to build retrievers
def build_retrievers(chroma):
    retriever_mmr = chroma.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    retriever_cos = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever_mmr, retriever_cos

# Retrieve documents function
def retrieve_docs(retriever, query):
    return retriever.get_relevant_documents(query)

def build_qa_models():
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=200,
        timeout=None,
        max_retries=2,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token()
    )

    chroma = load_chroma()

    # MMR retriever
    retriever_mmr = chroma.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )
    
    # Cosine Similarity retriever
    retriever_cos = chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Define RetrievalQA without passing `prompt_template` directly
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

# Function to retrieve documents without formatting the final prompt
def retrieve_docs(qa, query):
    st.info("Retrieving documents...")
    retrieved_docs = qa.retriever.get_relevant_documents(query)
    return retrieved_docs


# Function to save feedback locally in a JSON file
def save_feedback(query, answer, feedback, document_contents):
    try:
        # Log entry to be saved
        log_entry = {
            "query": query,
            "answer": answer,
            "feedback": feedback,
            "retriever_method": st.session_state.selected,
            "document_contents": [doc.page_content for doc in document_contents]
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

        st.success("Your feedback and document contents have been saved successfully!")

    except Exception as e:
        st.error(f"An error occurred while saving feedback: {e}")
        st.write(f"Error details: {e}")

# Main function
def main():
    st.title("Ask about Markdown Documents")

    # Load and initialize Chroma
    chroma = load_chroma()
    retriever_mmr, retriever_cos = build_retrievers(chroma)

    query = st.text_input("Query: ", key="input")

    if query:
        # Retrieve documents using both MMR and Cosine Similarity
        docs_mmr = retrieve_docs(retriever_mmr, query)
        docs_cos = retrieve_docs(retriever_cos, query)

        # Display retrieved documents
        col1, col2 = st.columns(2)
        with col1:
            display_documents(docs_mmr, "MMR Retrieval Results")
        with col2:
            display_documents(docs_cos, "Cosine Similarity Retrieval Results")

        # User chooses preferred retrieval result
        st.markdown("### Which retrieval result was better?")
        if st.button("MMR is better"):
            st.session_state.selected = "MMR"
            selected_docs = docs_mmr
        elif st.button("Cosine Similarity is better"):
            st.session_state.selected = "Cosine"
            selected_docs = docs_cos

        # Generate final answer if a choice was made
        if "selected" in st.session_state and st.session_state.selected:
            context = "\n\n".join([doc.page_content for doc in selected_docs])
            final_prompt = prompt_template.format(context=context)

            taide_llm = ChatOpenAI(
                model="TAIDE/a.2.0.0",
                temperature=0,
                max_tokens=200,
                timeout=None,
                max_retries=2,
                openai_api_base="https://td.nchc.org.tw/api/v1/",
                openai_api_key=get_token()
            )
            final_answer = taide_llm(final_prompt)

            # Display final answer
            st.markdown(f"## Final Answer (from {st.session_state.selected})")
            st.write(final_answer.content)

if __name__ == '__main__':
    main()
