import os
import sys
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

import os
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

def initialize_vectorstore():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENV"],
    )

    index_name = os.environ["PINECONE_INDEX"]
    # embeddings = OpenAIEmbeddings() # Priced
    embeddings = HuggingFaceEmbeddings()
    return pc.from_existing_index(index_name, embeddings)

if __name__ == "__main__":
    file_path = sys.argv[1]
    loader = UnstructuredFileLoader(file_path) 
    raw_docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_docs)

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)

