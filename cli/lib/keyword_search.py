import math
from .search_utils import clean_stop_words_and_stem,   DEFAULT_SEARCH_LIMIT, tokenize_text
from .InvertedIndex import InvertedIndex

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    index = InvertedIndex()
    index.load()

    processed_query = clean_stop_words_and_stem(tokenize_text(query))
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
    tokens = clean_stop_words_and_stem(tokenize_text(term))
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
    
    




        
    



