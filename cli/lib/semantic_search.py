import json
import pathlib
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from typing import Any, Callable


from .search_utils import format_search_result, get_movies

EMBEDDINGS_FILE = "movie_embeddings.npy"
CHUNK_EMBEDDINGS_FILE = "chunk_embeddings.npy"
CHUNK_METADATA_FILE = "chunk_metadata.json"

CWD = pathlib.Path.cwd()
CACHE_DIR = CWD / "cache"

EMBEDDINGS_PATH = CACHE_DIR / EMBEDDINGS_FILE
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / CHUNK_EMBEDDINGS_FILE
CHUNK_METADATA_PATH = CACHE_DIR / CHUNK_METADATA_FILE

MOVIES_DATASET = CWD / "data" / "movies.json" 

DEFAULT_SEMANTIC_OVERLAP = 1
DEFAULT_CHUNK_SIZE = 4



class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self,text: str) -> Any:
        if not text:
            raise ValueError("empty text")

        output = self.model.encode([text])
        return output[0]

    def build_embeddings(self,documents) -> np.ndarray:
        self.documents = documents
        str_reprs = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            str_reprs.append(f"{doc["title"]}:{doc["description"]}")
        self.embeddings = self.model.encode(str_reprs, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH,self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents) -> Any:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if pathlib.Path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)
    
    def search(self,query: str,limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded, load or create them first")
        
        query_embedding = self.generate_embedding(query)
        sim_results = []
        for doc_id, doc_embedding in enumerate(self.embeddings,1):
            cosine_sim = cosine_similarity(query_embedding,doc_embedding)
            sim_results.append((cosine_sim,doc_id))

        sim_results.sort(key=lambda x: x[0],reverse=True)
        search_results = []
        for res in  sim_results[:limit]:
            doc = self.document_map[res[1]]
            result = format_search_result(
                doc["id"],
                doc["title"],
                doc["description"],
                res[0]
            )
            search_results.append(result)
        return search_results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self,documents: Any) -> np.ndarray:
        self.documents = documents
        all_chunks = []
        metadata = []
        for i,doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            text = doc.get("description","")
            if not text.strip():
                continue
            chunks = semantic_chunk_text_command(
                text,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_SEMANTIC_OVERLAP,
            )
            
            for j,chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "movie_idx": i, "chunk_idx": j,
                    "total_chunks": len(chunks)
                })

        
        self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar=True)
        self.chunk_metadata = metadata

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH,self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH,"w") as file:       
            json.dump({"chunks": metadata, "total_chunks": len(metadata)},file,indent=2)

        return self.chunk_embeddings

    
    def load_or_create_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if pathlib.Path.exists(CHUNK_EMBEDDINGS_PATH) and pathlib.Path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH) as file:
                self.chunk_metadata = json.load(file)["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunks(self,query:str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is  None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first"
            )

        query_embeddings = self.generate_embedding(query)
        chunk_scores = []

        for i,chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embeddings,chunk_embedding)
            chunk_scores.append({
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score,
            })

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores 
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                 movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key= lambda x: x[1],reverse=True)
        
        results = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:100],
                    score=score,
                )
            )

        return results





def semantic_search_command(query: str,limit:int | None = None) -> list[dict]:
    sem_search = SemanticSearch()
    documents = get_movies()
    sem_search.load_or_create_embeddings(documents["movies"])
    if limit is not None:
        return sem_search.search(query,limit)
    else:
        return sem_search.search(query)

def semantic_search_chunked_command(query:str,limit:int | None = None) -> None:
    sem_search = ChunkedSemanticSearch()
    documents = []
    results = []
    documents = get_movies()
    sem_search.load_or_create_embeddings(documents["movies"])
    if limit is not None:
        results = sem_search.search_chunks(query,limit)
    else:
        results =  sem_search.search_chunks(query)

    for i,res in enumerate(results,1):
        print(f"\n{i}. {res["title"]} (score: {res["score"]})")
        print(f"   {res["document"]}...")



def verify_model() -> None:
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")

def embed_text(text: str) -> None:   
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    sem_search = SemanticSearch()
    documents = get_movies()
    embeddings = sem_search.load_or_create_embeddings(documents["movies"])
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> Any:
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def embed_chunks_command() -> None:
    chunk_semantic_search = ChunkedSemanticSearch()
    documents = get_movies()
    embeddings = chunk_semantic_search.load_or_create_embeddings(documents["movies"])
    print(f"Generated {len(embeddings)} chunked embeddings")
    

def cosine_similarity(vec1: list, vec2: list) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

    
def standard_chunk_text_command(text: str, chunk_size: int, overlap:int) -> list[str]:
    chunks = chunk_text(text,chunk_size,overlap,word_splitter)
    return chunks

def semantic_chunk_text_command(text:str, max_chunk_size: int, overlap:int) -> list[str]:
    chunks = chunk_text(text,max_chunk_size,overlap,sentence_splitter)
    return chunks


def chunk_text(text:str,chunk_size:int,overlap:int,splitter: Callable[[str], list[str]]) -> list[str]:
    chunks = []
    split_result = splitter(text)
    start = 0
    n = len(split_result)
    while start < n:
        chunk_parts = split_result[start : start + chunk_size]
        if chunks and len(chunk_parts) <= overlap:
            break
        content = " ".join(chunk_parts).strip()
        if not chunk_parts:
            continue
        chunks.append(content)
        start += chunk_size - overlap
    return chunks

def word_splitter(text:str) -> list[str]:
    result = text.split()
    return result

def sentence_splitter(text:str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    result = re.split(r"(?<=[.!?])\s+",text)
    if len(result) == 1 and not text.endswith((".", "!", "?")):
        result = [text]
    return result



