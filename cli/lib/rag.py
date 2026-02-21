from google.genai import  types

from .llm_prompts import LLM_RAG_QUESTION, LLM_RAG_RESPONSE, LLM_RAG_SUMMARIZE
from .hybrid_search import HybridSearch
from .search_utils import DEFAULT_SEARCH_LIMIT, RRF_K, load_movies


def rag_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT, prompt_instruction : str = LLM_RAG_RESPONSE) -> tuple[list[dict],str]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query,RRF_K,limit)
    
    results_as_str = "\n".join([f"- {res["title"]} - {res["document"]}" for res in results])
    prompt = f"Query: {query}\n{results_as_str}"
    response = hybrid_search.ai_client.models.generate_content(
        contents=prompt,
        model=hybrid_search.model,
        config=types.GenerateContentConfig(
            system_instruction=prompt_instruction
        )
    )

    assert response.text is not None

    return (results, response.text)



def summarize(query: str) -> tuple[list[dict],str]:
    return rag_command(query,DEFAULT_SEARCH_LIMIT,LLM_RAG_SUMMARIZE)

def question_command(query: str, limit : int = 5) -> tuple[list[dict],str]:
     return rag_command(query,limit,LLM_RAG_QUESTION)




