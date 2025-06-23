import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./data")

llm = ChatOpenAI(
    model_name=os.environ["OPENAI_API_MODEL"],
    temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),
    streaming=True,
)
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def chat_with_bot(prompt):
    response = rag_chain.invoke({"input": prompt})
    return response["answer"]

iface = gr.Interface(
    fn=chat_with_bot,
    inputs="text",
    outputs="text",
    title="RAG Chat bot",
    description="Ask anything and get responses based on retrieved context.",
)

if __name__ == "__main__":
    iface.launch()
