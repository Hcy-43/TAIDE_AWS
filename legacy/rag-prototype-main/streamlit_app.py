# custom code for the streamlit app
from local_hf import reader
from taide_chat import taide_llm

# dependencies for streamlit and langchain
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline

# dependencies for system
import os
import asyncio
import os
import time
import datetime
import json
from dotenv import load_dotenv

load_dotenv()

openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
    timeout=None,
    max_retries=2,
)

hf_llm = HuggingFacePipeline(pipeline=reader)

llm = taide_llm # change this use different LLM provider

rag_pipelines = [
    "multilingual-e5",
    "text_embedding_3_large",
    "text_embedding_3_small",
    "vanilla",
]


async def get_answer_multilingual_e5(query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "sinica-rag-test-0730-multilingual-e5-large"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=1)
    chain = load_qa_chain(llm, chain_type="map_reduce")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer


async def get_answer_text_embedding_3_large(query: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "sinica-rag-test-0730-text-embedding-3-large"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=1)
    chain = load_qa_chain(llm, chain_type="map_reduce")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer


async def get_answer_text_embedding_3_small(query: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = "sinica-rag-test-0730-text-embedding-3-small"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=1)
    chain = load_qa_chain(llm, chain_type="map_reduce")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer


async def get_answer_without_rag(query: str) -> str:
    chain = load_qa_chain(llm)
    answer = await asyncio.to_thread(chain.run, input_documents=[], question=query)
    return answer


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def display_answer():
    for i in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(i["question"])
        with st.chat_message("ai"):
            st.write(i["answers"])
        if "feedback" in i:
            with st.chat_message(avatar="👨‍💻", name="feedback"):
                st.write(i["feedback"])


async def create_answer(question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({
        "question": question,
        "answers": {
            "multilingual-e5": await get_answer_multilingual_e5(question),
            "text_embedding_3_large": await get_answer_text_embedding_3_large(question),
            "text_embedding_3_small": await get_answer_text_embedding_3_small(question),
            "vanilla": await get_answer_without_rag(question)
        },
        "message_id": len(st.session_state.chat_history),
    })


def store_feedback(data):
    current_time = time.time()
    current_time_readable = datetime.datetime.fromtimestamp(
        current_time).strftime('%Y-%m-%d_%H:%M:%S')

    filepath = os.path.join(os.path.dirname(__file__),
                            "logs", f"{current_time_readable}-log.json")
    # if folder does not exist, create it
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def fbcb():
    message_id = len(st.session_state.chat_history) - 1
    if message_id >= 0:

        st.session_state.chat_history[message_id]["feedback"] = {}
        for pipeline in rag_pipelines:
            if f'fb_k_{pipeline}' in st.session_state:
                st.session_state.chat_history[message_id]["feedback"][
                    pipeline] = st.session_state[f'fb_k_{pipeline}']
    display_answer()
    store_feedback(
        st.session_state.chat_history
    )


async def main():
    if question := st.chat_input(placeholder="Ask your question here .... !!!!"):
        await create_answer(question)
        display_answer()
        with st.form(f'feedback-form'):
            st.header("Feedback")

            for pipeline in rag_pipelines:
                st.write(pipeline)
                streamlit_feedback(align="flex-start", key=f'fb_k_{pipeline}', feedback_type="faces",
                                   optional_text_label="[Optional] Please provide an explanation",)
            st.form_submit_button('Save feedback', on_click=fbcb)
asyncio.run(main())
