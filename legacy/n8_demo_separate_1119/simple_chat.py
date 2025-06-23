import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import requests
# Load environment variables
load_dotenv()

# Page layout settings
st.set_page_config(page_title="Conversational AI Chatbot", layout="wide")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Add an expander to view the message contents in session state
view_messages = st.expander("View the message contents in session state")
with view_messages:
    if msgs.messages:
        for message in msgs.messages:
            st.write(f"**{message.type.capitalize()}**: {message.content}")
    else:
        st.write("No messages in session state.")

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
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="./azure_db")

def build_qa_chain(chroma):
    """Build a conversational QA chain with history support."""
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=200,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )

    # Prompt template for conversation with history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一個非常智能的助理。以下是用戶的問題和相關上下文，請根據這些內容回答問題。"),
            MessagesPlaceholder(variable_name="history"),  # History placeholder
            ("human", "{question}"),
        ]
    )

    # Create a chain combining prompt, LLM, and history
    retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chain = prompt | taide_llm

    # Enable history persistence
    history = StreamlitChatMessageHistory(key="chat_messages")
    qa_chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return qa_chain_with_history, history, retriever

def main():
    st.title("Conversational AI Chatbot")
    chroma = load_chroma()
    qa_chain_with_history, history, retriever = build_qa_chain(chroma)

    # Display chat history
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)

    # User input
    if user_query := st.chat_input("Ask your question:"):
        st.chat_message("human").write(user_query)

        # Retrieve documents as context
        retrieved_docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join(
            f"元數據：{doc.metadata}\n內容：{doc.page_content}" for doc in retrieved_docs
        )

        # Run the chain with the context and append the AI response to history
        response = qa_chain_with_history.invoke(
            {"context": context, "question": user_query},
            config={"session_id": "user_session"},
        )
        st.chat_message("ai").write(response.content)
    

if __name__ == "__main__":
    main()
