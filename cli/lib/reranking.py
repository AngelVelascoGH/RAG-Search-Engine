import json
import os
import sys
from time import sleep

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

from .llm_prompts import LLM_RRF_SEARCH_EVALUATION, LLM_SYSTEM_INSTRUCTION_RERANK_BATCH, LLM_SYSTEM_INSTRUCTION_RERANK_INDIVIDUAL


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"

def llm_rerank_individual(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    for res in documents:
        new_rank = 0
        response = client.models.generate_content(
            model=model,
            contents=f"Movie: {res["title"]}, Query: {query}",
            config=genai.types.GenerateContentConfig(
                system_instruction=LLM_SYSTEM_INSTRUCTION_RERANK_INDIVIDUAL
            )
        )
        if response.text is None:

            print(f"-- Gemini Content error --")
            print(response)
            print("---------------------------")
            sys.exit(1)

        try:
            new_rank = int(response.text)
        except ValueError:
            print(f"Different than a num: {response.text}")
            new_rank = res["score"]
    
        res["metadata"]["rerank_score"] = new_rank
        #to avoid rate limit on individual ranks
        sleep(1)
    results = sorted(documents,key= lambda x: x["metadata"]["rerank_score"], reverse=True)
    return results

def llm_rerank_batch(query:str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(f"id-{doc_id}: {doc.get("title","")} - {doc.get("document","")[:200]}")
    
    doc_list_str = "\n".join(doc_list)

    prompt = f"search query: {query}." + doc_list_str


    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=LLM_SYSTEM_INSTRUCTION_RERANK_BATCH
        )
    )
    ranking_text = (response.text or "").strip()

    parsed_reranked_ids = json.loads(ranking_text)

    reranked = []
    for i, doc_id in enumerate(parsed_reranked_ids,1):
        if doc_id in doc_map:
            doc = {**doc_map[doc_id]}
            doc["metadata"]["rerank_score"] = i
            reranked.append(doc)

    return reranked[:limit]

def cross_encoder(query:str, documents: list[dict], limit: int = 5) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get("title","")} - {doc.get("document","")}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for i,doc in enumerate(documents):
        doc["metadata"]["rerank_score"] = scores[i]

    documents = sorted(documents,key=lambda x: x["metadata"]["rerank_score"], reverse=True)

        

    return documents[:limit]
    
def llm_rerank(query:str, documents: list[dict], method: str = "batch", limit: int = 5) -> list[dict]:
    match method:
        case "individual":
            return llm_rerank_individual(query,documents,limit)
        case "batch":
            return llm_rerank_batch(query,documents,limit)
        case _:
            return cross_encoder(query,documents,limit)


def llm_evaluate_ranks(query : str, results : list[dict]) -> None:
    results_to_prompt = [f"{res["title"]} - {res["document"][:200]}" for res in results]
    prompt = f"Query: {query}\n{"\n".join(results_to_prompt)}"

    llm_response = client.models.generate_content(
        contents=prompt,
        model=model,
        config=genai.types.GenerateContentConfig(
            system_instruction=LLM_RRF_SEARCH_EVALUATION
        )
    )

    
    if llm_response.text is None:
        print(f"-- Gemini Content error --")
        print(llm_response)
        print("---------------------------")
        sys.exit(1)

    llm_evaluation = json.loads(llm_response.text)

    for i,result in enumerate(results,1):
        print(f"{i}. {result["title"]}: {llm_evaluation[i-1]}/3")
