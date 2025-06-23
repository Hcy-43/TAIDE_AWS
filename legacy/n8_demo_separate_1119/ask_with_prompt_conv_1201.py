import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import json
import requests

# Load environment variables
load_dotenv()

# Page layout settings
st.set_page_config(page_title="Conversational AI Chatbot", layout="wide")

# 
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

def get_token():
    """Retrieve TAIDE API token from environment variables."""
    username = os.getenv("TAIDE_EMAIL")
    password = "taidetaide"
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if r.status_code == 200:
        return r.json().get("access_token")
    else:
        raise ConnectionError(f"Failed to retrieve token. Status code: {r.status_code}")


def load_chroma():
    """Load the Chroma database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="./azure_db")

def build_qa_model(chroma):
    """Build QA model with conversational capability."""
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=200,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )

    # Define a custom prompt template in Traditional Chinese
    custom_prompt_template = PromptTemplate(
        input_variables=["context", "query"],  # Add "context" as required
        template="""
        你是一個非常智能的助理。以下是用戶的問題和相關上下文，請根據這些內容回答問題。

        上下文：
        {context}

        問題：
        {query}

        注意：
        - 你的回答必須引用上述的上下文內容，而不是自行生成答案。
        - 如果無法從上下文中找到答案，請直接回答「我不知道」或告知用戶資料不足，但請不要說因為不知道要回答什麼而亂回答。

        回答時請給出清晰且簡潔的回答。
        """,
    )

    # Create retriever
    retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Build RetrievalQA model
    qa_model = RetrievalQA.from_chain_type(
        llm=taide_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )

    return qa_model, custom_prompt_template


def save_feedback(query, final_answer, feedback, documents, retrieval_method):
    """Save feedback to a JSON file."""
    log_entry = {
        "query": query,
        "answer": final_answer,
        "feedback": feedback,
        "retrieval_method": retrieval_method,
        "document_contents": [doc.page_content for doc in documents],
    }
    file_path = "./feedback_log.json"
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    data.append(log_entry)
    with open(file_path, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    st.success("Feedback saved successfully!")


def main():
    st.title("Conversational AI Chatbot")
    chroma = load_chroma()
    qa_model, custom_prompt_template = build_qa_model(chroma)

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Step 1: User enters their query
    user_query = st.text_input("Ask your question:")
    if user_query:
        # Sanitize query
        user_query = str(user_query).replace("\n", " ").strip()

        # Append the user's query to the conversation history
        st.session_state.history.append({"role": "user", "content": user_query})

        # Format the conversation history as a string
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in st.session_state.history
        ) if st.session_state.history else ""

        # Step 2: Retrieve documents
        retrieved_docs = qa_model.retriever.invoke(user_query)  # Ensure user_query is a string

        # Format the context with metadata and content
        context = "\n\n".join(
            f"元數據：{doc.metadata}\n內容：{doc.page_content}" for doc in retrieved_docs
        )

        # Step 3: Generate the answer
        answer = qa_model.invoke({"query": user_query})

        # Append the bot's answer to the conversation history
        st.session_state.history.append({"role": "assistant", "content": answer["result"]})

        # Display the conversation
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")

        # Feedback system
        feedback = st.radio("How was the answer?", ["👍 Good", "👎 Not Good"], key=f"feedback_{len(st.session_state.history)}")
        if st.button("Submit Feedback", key=f"submit_feedback_{len(st.session_state.history)}"):
            save_feedback(user_query, answer["result"], feedback, retrieved_docs, "similarity")


if __name__ == "__main__":
    main()
