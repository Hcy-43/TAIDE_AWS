import os
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import json
import requests

load_dotenv()

# Function to get the TAIDE API token
def get_token():
    username = os.getenv("TAIDE_EMAIL")
    password = os.getenv("TAIDE_PASSWORD")
    r = requests.post(
        "https://td.nchc.org.tw/api/v1/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if r.status_code == 200:
        return r.json().get("access_token")
    else:
        raise ConnectionError(f"Failed to retrieve token. Status code: {r.status_code}")

# Extract text from PDF using Azure Document Intelligence
def extract_text_from_pdf_with_azure_di(pdf_path):
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=pdf_path,
        api_key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
        api_endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        api_model="prebuilt-layout"
    )
    docs = loader.load()
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
    return text_splitter.split_text(docs[0].page_content)

# Load all PDFs from the directory
def load_all_pdfs(directory="./docs"):
    pdf_texts = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            pdf_texts.append((file_name, extract_text_from_pdf_with_azure_di(os.path.join(directory, file_name))))
    return pdf_texts

# Load Chroma DB for embeddings
def load_chroma():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="./data")

# Build vector store from PDF texts
def build_vector_store(pdf_texts):
    chroma = load_chroma()
    for file_name, pdf_text in pdf_texts:
        text_content = [chunk.page_content for chunk in pdf_text]
        chroma.add_texts(text_content, metadatas=[{"source": file_name}] * len(text_content))
    return chroma

# Build QA retrieval models
def build_qa_models(chroma):
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=200,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )
    retriever_mmr = chroma.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    retriever_cos = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return (
        RetrievalQA.from_chain_type(llm=taide_llm, chain_type="stuff", retriever=retriever_mmr, return_source_documents=True),
        RetrievalQA.from_chain_type(llm=taide_llm, chain_type="stuff", retriever=retriever_cos, return_source_documents=True),
    )

# Save feedback to JSON
def save_feedback(query, final_answer, feedback, documents):
    log_entry = {
        "query": query,
        "answer": final_answer,
        "feedback": feedback,
        "document_contents": [doc.page_content for doc in documents],
    }
    file_path = "./feedback_log.json"
    data = []
    
    # ファイルが存在しているかチェック
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)  # JSONを読み込む
            except json.JSONDecodeError:
                data = []  # 空のファイルの場合は空リストを使用
    
    # 新しいフィードバックを追加
    data.append(log_entry)
    
    # JSONファイルに保存
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    st.success("Feedback saved successfully!")


# Main function
def main():
    st.title("Ask about PDF")
    pdf_texts = load_all_pdfs()
    if not pdf_texts:
        st.warning("No PDFs found in the 'docs' directory.")
        return

    chroma = build_vector_store(pdf_texts)
    qa_mmr, qa_cos = build_qa_models(chroma)

    query = st.text_input("Enter your query:")
    if query:
        answer_mmr = qa_mmr.invoke(query)
        answer_cos = qa_cos.invoke(query)

        # Display results for MMR
        st.subheader("MMR Retrieval Results")
        for doc in answer_mmr["source_documents"]:
            st.write(doc.page_content)
            st.write("---")

        # Display results for Cosine Similarity
        st.subheader("Cosine Similarity Retrieval Results")
        for doc in answer_cos["source_documents"]:
            st.write(doc.page_content)
            st.write("---")

        # Let the user choose which retrieval method was better
        st.subheader("Final Answer")
        selected_method = st.radio("Choose the best retrieval method:", ["MMR", "Cosine Similarity"])

        if selected_method == "MMR":
            final_answer = answer_mmr["result"]
            selected_docs = answer_mmr["source_documents"]
        else:
            final_answer = answer_cos["result"]
            selected_docs = answer_cos["source_documents"]

        st.write(f"### Final Answer ({selected_method} Retrieval)")
        st.write(final_answer)

        # Feedback system
        feedback = st.radio("How was the answer?", ["👍 Good", "👎 Not Good"])
        if st.button("Submit Feedback"):
            save_feedback(query, final_answer, feedback, selected_docs)

if __name__ == '__main__':
    main()
