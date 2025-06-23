# pip install pycryptodome
from glob import glob
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
from langchain.chains import RetrievalQA # Deprecated 
from langchain.chains import create_retrieval_chain
load_dotenv()

def init_page():
    st.set_page_config(
        page_title="Ask about PDF",
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []


def get_pdf_text():
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
        return text_splitter.split_text(text)
    else:
        return None


def load_chroma():
    return Chroma(
        embedding_function=HuggingFaceEmbeddings(),
        persist_directory="./data"
    )


def build_vector_store(pdf_text):
    chroma = load_chroma()
    chroma.add_texts(pdf_text)


def build_qa_model(llm):
    chroma = load_chroma()
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
    )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)

    return answer, cb.total_cost


def page_ask_my_pdf():
    st.title("Ask about PDF")

    llm = ChatOpenAI(
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        model_name=os.environ["OPENAI_API_MODEL"],)
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("Typing answer..."):
                    answer, cost = ask(qa, query)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer["result"])
                st.markdown("## Source Documents")
                displayed_documents = set()
                for doc in answer["source_documents"]:
                    if doc.page_content not in displayed_documents:
                        st.write(doc.page_content)
                        displayed_documents.add(doc.page_content)



def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask about PDF"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask about PDF":
        page_ask_my_pdf()

if __name__ == '__main__':
    main()
