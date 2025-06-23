import pandas as pd
import asyncio
from typing import List
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from dotenv import load_dotenv

from eval import compute_context_recall_score, compute_faithfulness_score

# Load .env
load_dotenv()

# Initialize Pinecone index and embeddings model
embedding_model_name = "text-embedding-3-large" 
index_name = "sinica-rag-test-0730-text-embedding-3-large"  

embeddings = PineconeEmbeddings(model=embedding_model_name)
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

async def retrieve_context_from_pinecone(query: str) -> List[str]:
    """Retrieve context from Pinecone for the given query."""
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=1)
    contexts = [doc.page_content for doc in docs]
    return contexts

async def process_csv_and_compute_scores(csv_file: str):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Assuming the CSV has columns 'Question', 'Answer'
    questions = df['Question'].tolist()
    answers = df['Answer'].tolist()

    # Retrieve contexts from Pinecone
    contexts = []
    for question in questions:
        context = await retrieve_context_from_pinecone(question)
        contexts.append(context)

    # Compute context recall score
    context_recall_scores = compute_context_recall_score(contexts, answers, questions)
    
    # Compute faithfulness score
    faithfulness_scores = compute_faithfulness_score(contexts, answers, questions)

    # Return the scores as a DataFrame
    scores_df = pd.DataFrame({
        'Question': questions,
        'Context_Recall_Score': context_recall_scores,
        'Faithfulness_Score': faithfulness_scores
    })
    
    return scores_df

if __name__ == "__main__":
    csv_file_path = "dataset/qa-pair/MedQuAD-QA-pair.csv"
    scores_df = asyncio.run(process_csv_and_compute_scores(csv_file_path))
    
    print(scores_df)
    scores_df.to_csv('evaluation_scores.csv', index=False)

