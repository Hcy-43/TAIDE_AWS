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
st.set_page_config(page_title="Ask about Markdown", layout="wide")

def get_token():
    """Retrieve TAIDE API token from environment variables."""
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

def load_chroma():
    """Load the Chroma database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="./azure_db")

def build_qa_models(chroma):
    """Build QA models for MMR and Cosine Similarity retrieval."""
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=200,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )

    # Define a custom prompt template in Traditional Chinese
    custom_prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        你是一個非常智能的助理。以下是根據問題提供的上下文內容，請根據這些內容回答問題。

        上下文：
        {context}

        問題：
        {question}

        注意：
        - 你的回答必須引用上述的上下文內容，而不是自行生成答案。
        - 如果無法從上下文中找到答案，請直接回答「我不知道」或告知用戶資料不足，但請不要說因為不知道要回答什麼而亂回答。

        回答時請給出清晰且簡潔的回答。
        """
    )

    # Create retrievers
    retriever_mmr = chroma.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    retriever_cos = chroma.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Build RetrievalQA models with the custom prompt
    qa_mmr = RetrievalQA.from_chain_type(
        llm=taide_llm,
        chain_type="stuff",
        retriever=retriever_mmr,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )

    qa_cos = RetrievalQA.from_chain_type(
        llm=taide_llm,
        chain_type="stuff",
        retriever=retriever_cos,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )

    return qa_mmr, qa_cos, custom_prompt_template

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
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    data.append(log_entry)
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    st.success("Feedback saved successfully!")

def refine_query_with_taide(user_query):
    """
    Refines the user's query by sending it to TAIDE for rephrasing into a more precise query.
    """
    taide_llm = ChatOpenAI(
        model="TAIDE/a.2.0.0",
        temperature=0,
        max_tokens=50,
        openai_api_base="https://td.nchc.org.tw/api/v1/",
        openai_api_key=get_token(),
    )

    # Define the refinement prompt
    refinement_prompt = f"""
        你是一個對LLM熟悉的助理。以下是對LLM不熟悉的用戶的提問內容，可能過度簡潔，或者過度複雜，
        讓LLM混亂，或著讓LLM不知道要檢索什麼樣的內容。
        請依照用戶原始提問，生成一個對LLM好理解的query：

        用戶原始的提問：
        {user_query}

        請只提供優化後的query，不要回傳其他任何東西，包括原始的query，以免影響到後續的處理。
        請用一句優化後的query回傳。
    """

    # Send the prompt to TAIDE and extract the content of the response
    refined_query = taide_llm(refinement_prompt)  # Returns a ChatCompletion
    return refined_query.content.strip()  # Access the content of the response



def main():
    st.title("醫療聊天機器人")
    chroma = load_chroma()
    qa_mmr, qa_cos, custom_prompt_template = build_qa_models(chroma)

    # Step 1: User enters their query
    user_query = st.text_input("Enter your query:")
    if user_query:
        # Step 2: Refine the query using TAIDE
        with st.spinner("Refining your query..."):
            refined_query = refine_query_with_taide(user_query)
        st.success(f"Refined Query: {refined_query}")

        # Step 3: Retrieve documents using the refined query
        retrieved_mmr = qa_mmr.retriever.get_relevant_documents(refined_query)
        retrieved_cos = qa_cos.retriever.get_relevant_documents(refined_query)

        # Step 4: Format the context with metadata and content
        context_mmr = "\n\n".join(
            f"元數據：{doc.metadata}\n內容：{doc.page_content}" for doc in retrieved_mmr
        )
        context_cos = "\n\n".join(
            f"元數據：{doc.metadata}\n內容：{doc.page_content}" for doc in retrieved_cos
        )

        # Step 5: Generate prompts with metadata included
        prompt_mmr = custom_prompt_template.format(context=context_mmr, question=refined_query)
        prompt_cos = custom_prompt_template.format(context=context_cos, question=refined_query)

        # Step 6: Generate answers
        answer_mmr = qa_mmr.invoke(refined_query)
        answer_cos = qa_cos.invoke(refined_query)

        # Step 7: Display results
        st.markdown("### Comparison of Retrieval Methods")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### MMR Retrieval")
            # st.markdown("**Retrieved Documents**")
            # for doc in retrieved_mmr:
            #     st.markdown(f"**Source**: {doc.metadata.get('source', 'Unknown')}")
            #     st.write(doc.page_content)
            #     st.write("---")
            # st.markdown("**Generated Prompt**")
            # st.code(prompt_mmr, language="markdown")
            st.markdown("**Generated Answer**")
            st.write(answer_mmr["result"])

        with col2:
            st.markdown("#### Cosine Similarity Retrieval")
            # st.markdown("**Retrieved Documents**")
            # for doc in retrieved_cos:
            #     st.markdown(f"**Source**: {doc.metadata.get('source', 'Unknown')}")
            #     st.write(doc.page_content)
            #     st.write("---")
            # st.markdown("**Generated Prompt**")
            # st.code(prompt_cos, language="markdown")
            st.markdown("**Generated Answer**")
            st.write(answer_cos["result"])

        # Let the user choose which retrieval method was better
        st.markdown("### Final Answer Selection")
        selected_method = st.radio("Choose the best retrieval method:", ["MMR", "Cosine Similarity"])
        final_answer = answer_mmr["result"] if selected_method == "MMR" else answer_cos["result"]
        selected_docs = retrieved_mmr if selected_method == "MMR" else retrieved_cos

        # Feedback system
        feedback = st.radio("How was the answer?", ["👍 Good", "👎 Not Good"])
        if st.button("Submit Feedback"):
            save_feedback(user_query, final_answer, feedback, selected_docs, selected_method)


if __name__ == "__main__":
    main()
