import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


from .reranking import llm_rerank, llm_evaluate_ranks

from .llm_prompts import LLM_SYSTEM_INSTRUCTION_EXPAND, LLM_SYSTEM_INSTRUCTION_REWRITE, LLM_SYSTEM_INSTRUCTION_SPELL

from .search_utils import  RRF_K, SEARCH_MULTIPLIER, format_search_result, get_movies

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        self.ai_client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self,query:str,alpha:float= 0.5, limit:int =5) -> list[dict]:
        results_bm25 = self._bm25_search(query=query,limit=limit * 500)
        results_semantic = self.semantic_search.search_chunks(query=query,limit=limit * 500)

        scores_bm25 = normalize_scores([score["score"] for score in results_bm25])
        scores_semantic = normalize_scores([score["score"] for score in results_semantic])
        documents_map = {}
        for doc in self.documents:
            documents_map[doc["id"]] = {"document":doc}
            
        for i in range(len(results_bm25)):
            results_bm25_doc_id = results_bm25[i].get("id")
            results_semantic_doc_id = results_semantic[i].get("id")
            documents_map[results_bm25_doc_id]["bm_score"] = scores_bm25[i]
            documents_map[results_semantic_doc_id]["sem_score"] = scores_semantic[i]

        for doc_id, doc in documents_map.items():
            sem_score = doc.get("sem_score",0)
            bm_score = doc.get("bm_score",0)
            documents_map[doc_id]["hybrid_score"] = alpha * bm_score + (1 - alpha) * sem_score

        sortedResults = sorted(documents_map.values(),key=lambda x: x["hybrid_score"],reverse=True)
        return sortedResults


    def rrf_search(self,query:str,k:int=60,limit:int=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]
        


    

def normalize_scores(scores: list[float]) -> list[float]:
    results = []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0 * len(scores)]

    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        results.append(normalized_score)


    return results

def rrf_score(rank:int, k:int = RRF_K) -> float:
    return 1 / (k + rank)

def reciprocal_rank_fusion(
    bm25_results: list[dict], semantic_results: list[dict], k: int = RRF_K
) -> list[dict]:
    rrf_scores = {}
    
    for rank, result in enumerate(bm25_results,start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank":None,
            }

        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank,k)

    for rank, result in enumerate(semantic_results,start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank":None,
            }

        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank,k)

    sorted_results = sorted(
        rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
    )

    rrf_results = []
    for doc_id, data in sorted_results:
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return rrf_results

def hybrid_search(query:str,alpha:float,limit:int) -> None:
    movies = get_movies()
    hybrid_search = HybridSearch(movies["movies"])

    res = hybrid_search.weighted_search(query,alpha,limit)

    for i,r in enumerate(res[:limit],1):
        print(f"{i}. {r["document"]["title"]}")
        print(f"Hybrid Score: {r["hybrid_score"]:.4f}")
        print(f"BM25: {r["bm_score"]:.4f}, Semantic: {r["sem_score"]:.4f}")
        print(f"{r["document"]["title"][:100]}...")

def rrf_search(query:str,k:int,limit:int,enhance:str,rerank:str,evaluate: bool) -> None:
    movies = get_movies()
    hybrid_search = HybridSearch(movies["movies"])
    
    query = enhanceQuery(hybrid_search,query,enhance)
    
    search_limit = limit * SEARCH_MULTIPLIER if rerank else limit
    res = hybrid_search.rrf_search(query,k,search_limit)

    if rerank:
        res = llm_rerank(query,res,rerank,limit)


    for i,r in enumerate(res,1):
        metadata = r["metadata"]
        print(f"{i}. {r["title"]}")
        if rerank:
            print(f"Rerank Score: {metadata["rerank_score"]}")
        print(f"RRF Score: {metadata["rrf_score"]:.4f}")
        print(f"BM25 Rank: {metadata["bm25_rank"]}, Semantic Rank: {metadata["semantic_rank"]}")
        print(f"{r["document"][:100]}...")

    llm_evaluate_ranks(query, res)


def enhanceQuery(hybrid_search: HybridSearch, query: str, operation:str) -> str:
    if not operation:
        return query

    system_prompt = ""
    
    match operation:
        case "spell":
            system_prompt = LLM_SYSTEM_INSTRUCTION_SPELL
        case "rewrite":
            system_prompt = LLM_SYSTEM_INSTRUCTION_REWRITE
        case "expand":
            system_prompt = LLM_SYSTEM_INSTRUCTION_EXPAND


    response = hybrid_search.ai_client.models.generate_content(
        model=hybrid_search.model,
        contents=query,
        config=types.GenerateContentConfig(
        system_instruction=system_prompt
        )
    )
    if response.text is not None:
        print(f"Enhanced query ({operation}): '{query}' -> '{response.text}'\n")
        return response.text
    else:
        print(f"Query could not be enhanced, error with Gemini API")
        return query

