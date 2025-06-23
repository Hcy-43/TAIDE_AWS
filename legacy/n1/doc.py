import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import Pinecone
# from langchain_community.embeddings import HuggingFaceEmbeddings # deprecated
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(embedding_function = embeddings, persist_directory = "./data")

# test html page https://management.ntu.edu.tw/IM/board/detail/sn/18391
file_path = "test.html"
loader = UnstructuredFileLoader(file_path)
raw_docs = loader.load()
text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(raw_docs)
vectorstore.add_documents(docs)

# Run with a command "python doc.py" in the terminal