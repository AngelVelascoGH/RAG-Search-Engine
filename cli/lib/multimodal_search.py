import os
from typing import Any

from dotenv import load_dotenv
from google import genai
import numpy as np

from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import cosine_similarity



from .search_utils import CACHE_DIR, PROJECT_ROOT, load_movies


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"



class MultimodalSearch():
    def __init__(self,documents: list[dict],model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.embeddings_path = os.path.join(CACHE_DIR,"text_embeddings.npy")
        self.embeddings = self.load_or_create_embeddings()

    def build_embeddings(self) -> Any:
        texts = [f"{doc["title"]} : {doc["description"]}" for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(self.embeddings_path,embeddings)
        return embeddings
    
    def load_or_create_embeddings(self) -> Any:
        if not os.path.exists(self.embeddings_path):
            return self.build_embeddings()

        embeddings = np.load(self.embeddings_path)
        if len(embeddings) != len(self.documents):
            embeddings = self.build_embeddings()

        return embeddings


    def search_with_image(self,imagePath: str) -> list[dict]:
        embedding = self.embed_image(imagePath)
        results = []
        for i,doc in enumerate(self.documents):
            cos_sim = cosine_similarity(embedding,self.embeddings[i])
            results.append(
                {
                    "id":i,
                    "title":doc["title"],
                    "description":doc["description"],
                    "sim_score":cos_sim,
                }
            )

        results.sort(key=lambda x: x["sim_score"], reverse=True)
        return results



    def embed_image(self,imagePath: str) -> Any:
        image = Image.open(imagePath)
        embeddings = self.model.encode([image])[0]
        return embeddings


def image_search_command(imagePath:str) -> None:
    fullImagePath = os.path.join(PROJECT_ROOT, imagePath)
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents)
    results = multimodal_search.search_with_image(fullImagePath)
    for i,res in enumerate(results[:5],1):
        print("--------------------------------------------------")
        print(f"{i}. {res["title"]} (similarity: {res["sim_score"]:.3f})")
        print(f"{res["description"][:200]}...")
        print("--------------------------------------------------\n")


        


