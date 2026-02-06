import collections
import math
import pathlib
import pickle

from nltk import defaultdict
from .search_utils import clean_stop_words_and_stem, format_search_result, get_movies, tokenize_text, DEFAULT_SEARCH_LIMIT

INDEX_FILE = "index.pkl"
DOCMAP_FILE = "docmap.pkl"
FREQ_FILE = "term_frequencies.pkl"
DOC_LENGTH_FILE = "doc_lengths.pkl"


CWD = pathlib.Path.cwd()
CACHE_DIR = CWD / "cache"
INDEX_PATH = CACHE_DIR / INDEX_FILE
DOCMAP_PATH = CACHE_DIR / DOCMAP_FILE
FREQ_PATH = CACHE_DIR / FREQ_FILE
DOC_LENGTH_PATH = CACHE_DIR / DOC_LENGTH_FILE

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int,dict] = {}
        self.term_frequencies : dict[int,collections.Counter] = {}
        self.doc_lengths : dict[int,int] = {}

    def __add_document(self,doc_id,text) -> None:
        tokens = clean_stop_words_and_stem(tokenize_text(text))
        self.doc_lengths[doc_id] = len(tokens)

        self.term_frequencies[doc_id] = collections.Counter(tokens)

        for t in tokens:
            self.index[t].add(doc_id)


    def get_documents(self,term) -> list[int]:
        doc_ids = self.index.get(term.lower(),set())
        return sorted(list(doc_ids))

    def get_tf(self,doc_id,term) -> int:
        token = clean_stop_words_and_stem(tokenize_text(term))

        if len(token) != 1:
            raise Exception("more than one token", token)

        return self.term_frequencies[doc_id][term]


    def build(self) -> None:
        movies = get_movies()
        for movie in movies["movies"]:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"],f"{movie["title"]} {movie["description"]}")

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

        with open(INDEX_PATH,'wb') as file: 
            pickle.dump(self.index,file)
        
        with open(DOCMAP_PATH,'wb') as file: 
            pickle.dump(self.docmap,file)

        with open(FREQ_PATH,'wb') as file: 
            pickle.dump(self.term_frequencies,file)

        with open(DOC_LENGTH_PATH,'wb') as file: 
            pickle.dump(self.doc_lengths,file)


    def load(self) -> None:

        if not DOCMAP_PATH.exists() or not INDEX_PATH.exists():
            raise FileNotFoundError("Cache files not found")

        with open(INDEX_PATH,'rb') as file:
            self.index = pickle.load(file)

        with open(DOCMAP_PATH,'rb') as file:
            self.docmap = pickle.load(file)

        with open(FREQ_PATH,'rb') as file:
            self.term_frequencies = pickle.load(file)

        with open(DOC_LENGTH_PATH,'rb') as file:
            self.doc_lengths = pickle.load(file)

    def bm25(self,doc_id: int, term: str) -> float:

        bm25_tf = self.get_bm25_tf(doc_id,term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self,query: str, limit: int) -> list[dict]:
        tokens = clean_stop_words_and_stem(tokenize_text(query))
        scores = {}

        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id,token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(),key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score = score
            )
            results.append(formatted_result)

        return results
            



    def get_bm25_idf(self, term: str) -> float:
        token = clean_stop_words_and_stem(tokenize_text(term))
        if len(token) != 1:
            raise Exception("term with multiple tokens")

        df = len(self.index[token[0]])

        bm25_idf = math.log((len(self.docmap) - df + 0.5)/(df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self,doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:

        token = clean_stop_words_and_stem(tokenize_text(term))
        if len(token) != 1:
            raise Exception("term with multiple tokens")

        avg_lenght = self.__get_avg_doc_length()
        if avg_lenght == 0:
            raise Exception("no documents, or no content")

        lenght_norm = 1 - b + b * (self.doc_lengths.get(doc_id,0) / self.__get_avg_doc_length())


        raw_freq = self.get_tf(doc_id,token[0])
        bm25_saturation = (raw_freq * (k1 + 1)) / (raw_freq + k1 * lenght_norm)
        return bm25_saturation

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths.keys()) == 0:
            return 0.0

        total_lenght = 0
        for doc in self.doc_lengths.keys():
            total_lenght += self.doc_lengths[doc]

        return total_lenght / len(self.doc_lengths.keys())



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
    
    




        
    



