import json

import pathlib
import string
from typing import Any

from nltk.stem import PorterStemmer
from .stop_words import get_stop_words

root = pathlib.Path.cwd()
movies_dataset = root / "data" / "movies.json" 
DEFAULT_SEARCH_LIMIT = 5

def get_movies() -> dict:
    with open(movies_dataset,"r") as file:
        movies = json.load(file)
        return movies

def clean_str(input: str) -> str:
    transtab_punctuation = str.maketrans("","",string.punctuation)
    input = input.lower()
    input = input.translate(transtab_punctuation)
    return input

def tokenize_text(text: str) -> list[str]:
    text = clean_str(text)
    tokens = [token for token in text.split() if token]
    return tokens

def clean_stop_words_and_stem(query: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stop_words = get_stop_words()
    clean_query = []
    for token in query:
        if token not in stop_words:
            clean_query.append(stemmer.stem(token))
    return clean_query

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, 2),
        "metadata": metadata if metadata else {},
    }
