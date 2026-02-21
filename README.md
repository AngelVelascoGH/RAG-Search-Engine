# RAG Search Engine
A Retrieval-Augmented Generation search engine for a movie dataset.

The project explores various search architectures, from traditional keyword matching to modern AI search using Google Gemini for LLM-added functionality(rerank,evaluation,rewrite, etc), or local models for semantic search.


## Project Overview

The system is designed to demonstrate the evolution of search technology, starting with basic retrieval and moving toward complex, context-aware systems that synthesize information from text and images.

## Features

### Search Architectures
- **Keyword Search**: Implements traditional term-based retrieval.
- **Semantic Search**: Uses vector embeddings to find relevant content based on meaning rather than just keywords.
- **Hybrid Search**: Combines keyword and semantic results to provide a balanced retrieval mechanism.

### Advanced RAG Components
- **Augmented Generation**: Integrates retrieved context into LLM prompts to reduce hallucinations and provide grounded answers.
- **Query Enhancement**: Rewrites user queries to improve retrieval performance.
- **Reranking**: Scores and reorders search results to ensure the most relevant information is prioritized.
- **Evaluation**: Tools for measuring the effectiveness of the search results and the quality of generated answers.

## Repository Structure

- `cli/`: Executable scripts for interacting with the search system.
- `cli/lib/`: Core logic and utility functions for search, ranking, and AI integration.
- `data/`: Sample data used for indexing and testing, the movies dataset used is in /exampleData 

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Google Gemini API Key (Free tier works with gemini-2.5-flash)

### Setup
1. Clone the repository.
2. Set your API key in your environment variables:
   ```bash
   export GEMINI_API_KEY='your_key_here'
3. Use the movies dataset, or a similar one (the structure must be the same)
4. run:
```bash
uv sync
```

### Usage Examples

```bash
uv run cli/augmented_generation_cli.py --query "Funny bear movie"
```

```bash
uv run cli/multimodal_search_cli.py image_search {relative image path}
```

Both of these commands should Return a list with the search results.
There is multiple commands created to test and evaluate the functionality of each search implementation, run:
```bash
uv run cli/{search_type_cli.py file}
```
to see the possible commands


Local Embeddings and frequency indexes are stored in /cache on the first run, this will take some time
