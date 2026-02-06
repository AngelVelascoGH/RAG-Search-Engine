import json
import pathlib
import numpy as np

from sentence_transformers import SentenceTransformer
from typing import Any

from sentence_transformers.util import semantic_search

EMBEDDINGS_FILE = "movie_embeddings.npy"
CWD = pathlib.Path.cwd()
CACHE_DIR = CWD / "cache"
EMBEDDINGS_PATH = CACHE_DIR / EMBEDDINGS_FILE
MOVIES_DATASET = CWD / "data" / "movies.json" 


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
    documents = []
    with open(MOVIES_DATASET,'r') as file:
        documents = json.load(file)
    embeddings = sem_search.load_or_create_embeddings(documents["movies"])
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> Any:
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: list, vec2: list) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

    

    



    
