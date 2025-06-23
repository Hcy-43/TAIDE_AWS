import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Function to load Chroma vector store
def load_chroma():
    return Chroma(
        embedding_function=HuggingFaceEmbeddings(),
        persist_directory="/data"  # Path to your Chroma vector store
    )

# Function to view elements (chunks) in the Chroma DB
def view_chroma_db_elements():
    # Load Chroma DB
    chroma = load_chroma()
    
    # Retrieve stored documents (chunks)
    stored_documents = chroma.get_texts()

    # Check if there are any chunks stored
    if stored_documents:
        print(f"Total Chunks Found: {len(stored_documents)}")
        for i, chunk in enumerate(stored_documents):
            print(f"\nChunk {i+1}:")
            print(chunk)
    else:
        print("No chunks found in the database.")

# Main function to execute the script
if __name__ == "__main__":
    print("Loading and viewing Chroma vector store contents...\n")
    view_chroma_db_elements()
