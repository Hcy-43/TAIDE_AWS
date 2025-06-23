import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests

# Load environment variables
load_dotenv()

# Page layout settings
st.set_page_config(page_title="Conversational RAG Chatbot with Summary", page_icon="📖")
st.title("📖 Conversational RAG Chatbot with Summary")

"""
A conversational chatbot powered by TAIDE and Retrieval-Augmented Generation (RAG). 
Use the **FINISH CONVERSATION** button to generate a summary of the conversation.
"""

# Function to retrieve TAIDE API token
def get_taide_token():
    username = os.getenv("TAIDE_EMAIL")
    password = "taidetaide"  # Replace this with the actual password
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if r.status_code == 200:
        return r.json().get("access_token")
    else:
        st.error("Failed to retrieve TAIDE token. Check your credentials.")
        st.stop()

# Set up the TAIDE LLM
taide_token = get_taide_token()
taide_llm = ChatOpenAI(
    model="TAIDE/a.2.0.0",
    temperature=0,
    max_tokens=200,
    openai_api_base="https://td.nchc.org.tw/api/v1/",
    openai_api_key=taide_token,
)

# Load Chroma retriever
def load_chroma():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="./azure_db")

retriever = load_chroma().as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Add an expander to view the message contents in session state
view_messages = st.expander("View the message contents in session state")
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```
    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)

# Set up the LangChain prompt with message history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一個非常智能的助理，請根據用戶的問題進行對話並給出清晰回答。"),
        MessagesPlaceholder(variable_name="history"),  # History placeholder
        ("human", "{question}"),
    ]
)

# Combine the prompt and TAIDE LLM with history
chain = prompt | taide_llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Handle user input
if prompt := st.chat_input("Ask your question:"):
    st.chat_message("human").write(prompt)

    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(prompt)

    # Display retrieved chunks
    st.subheader("Retrieved Chunks")
    for idx, doc in enumerate(retrieved_docs, start=1):
        st.markdown(f"**Chunk {idx}:**\n{doc.page_content}")

    # Incorporate retrieved context into the chain
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = chain_with_history.invoke({"question": prompt, "context": context}, config={"session_id": "any"})
    
    # Display the chatbot's response
    st.chat_message("ai").write(response.content)

# Generate summary when FINISH CONVERSATION is pressed
if st.button("FINISH CONVERSATION"):
    # Create a summary prompt from the conversation history
    conversation_history = "\n".join(
        f"{msg.type.capitalize()}: {msg.content}" for msg in msgs.messages
    )
    
    summary_prompt = f"""
    以下是整段對話記錄，請總結出這次對話的主要內容：
    
    對話記錄：
    {conversation_history}
    
    請以簡潔的方式總結此對話。
    """
    
    # Generate the summary using TAIDE
    summary_response = taide_llm.predict(summary_prompt)
    
    # Display the summary
    st.subheader("Conversation Summary")
    st.write(summary_response)
