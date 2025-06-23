import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI  # For TAIDE
import requests

# Load environment variables
load_dotenv()

# Page layout settings
st.set_page_config(page_title="StreamlitChatMessageHistory with TAIDE", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory with TAIDE")

"""
A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation with TAIDE.
The messages are stored in Session State across re-runs automatically.
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

    # Invoke the chain and append the response
    response = chain_with_history.invoke({"question": prompt}, config={"session_id": "any"})
    st.chat_message("ai").write(response.content)
