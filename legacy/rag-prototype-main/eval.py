from typing import List
from datasets import Features, Value, Sequence, Dataset
from ragas.metrics import context_recall, faithfulness
from ragas import evaluate
import pandas as pd

def compute_context_recall_score(context: List[List[str]], ground_truth: List[str], question: List[str]) -> pd.DataFrame:
    """
    Compute context recall scores for a given dataset.
    """
    
    data_samples = { 'contexts': context, 'ground_truth': ground_truth, 'question': question}
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[context_recall])
    df_score = score.to_pandas()
    
    return df_score

def compute_faithfulness_score(context: List[List[str]], answer: List[str], question: List[str]) -> pd.DataFrame:
    """
    Compute faithfulness scores for a given dataset.
    """
    
    data_samples = { 'contexts': context, 'answer': answer, 'question': question}
    dataset = Dataset.from_dict(data_samples)
    score = evaluate(dataset, metrics=[faithfulness])
    df_score = score.to_pandas()
    
    return df_score
