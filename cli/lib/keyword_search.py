import math
import os
import pickle
from collections import Counter, defaultdict


from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    tokenize_text,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)

    def bm25(self, doc_id: int, term: str) -> float:
        tf_component = self.get_bm25_tf(doc_id, term)
        idf_component = self.get_bm25_idf(term)
        return tf_component * idf_component

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = tokenize_text(query)

        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    index = InvertedIndex()
    index.load()

    processed_query = tokenize_text(query)
    seen, results = set(), []
    for token in processed_query:
        docs = index.get_documents(token)
        for doc_id in docs:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = index.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def tf_command(doc_id: int,term: str) -> None:
    index = InvertedIndex()
    index.load()
    freq = index.get_tf(doc_id,term)
    print(freq)

def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise Exception("0 or more terms were introduced")

    total_term = 0
    total_docs = 0

    #ammount of docs where term is in, not the total freq
    for doc in index.docmap.keys():
        total_docs += 1
        if term in index.term_frequencies[doc]:
            total_term += 1

    term_idf = math.log((total_docs + 1) / (total_term + 1))
    return term_idf


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    tf = index.get_tf(doc_id,term)
    idf = idf_command(term)
    return tf * idf

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    bm25_idf = index.get_bm25_idf(term)
    return bm25_idf

def bm25_tf_command(doc_id: int, term: str, k1: float | None = None, b: float | None = None) -> float:
    index = InvertedIndex()
    index.load()
    if k1:
        bm25_tf = index.get_bm25_tf(doc_id,term,k1)
    else:
        bm25_tf = index.get_bm25_tf(doc_id,term)

    return bm25_tf

def bm25_search_command(query: str, limit: int = 5) -> list[dict]:
    index = InvertedIndex()
    index.load()
    return index.bm25_search(query,limit)
