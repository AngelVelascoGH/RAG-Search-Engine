import json
import os
from dotenv import load_dotenv

from google import genai

from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch
from .search_utils import load_golden_dataset, load_movies

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"

def calculate_precision_and_recall(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> tuple[float,float]:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1


    return (relevant_count / k, relevant_count / len(relevant_docs))

def calculate_f1_score(precision:float, recall:float) -> float:
    return 2 * (precision * recall) / (precision + recall)


def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0

    retrieved_results = {}
    
    for test in test_cases:
        query = test["query"]
        relevant_docs = set(test["relevant_docs"])
        search_res = hybrid_search.rrf_search(query, 60, limit)
        retrieved_docs = []
        for res in search_res:
            title = res.get("title","")
            if title:
                retrieved_docs.append(title)

        precision,recall = calculate_precision_and_recall(retrieved_docs, relevant_docs, limit)
        f1_score = calculate_f1_score(precision,recall)

        retrieved_results[query] = {
            "precision":precision,
            "recall":recall,
            "f1_score":f1_score,
            "retrieved":retrieved_docs[:limit],
            "relevant":list(relevant_docs),
        }

        total_precision += precision

    return {
        "test_cases":len(test_cases),
        "limit": limit,
        "results": retrieved_results,
    }
