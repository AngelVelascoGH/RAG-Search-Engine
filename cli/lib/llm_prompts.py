LLM_SYSTEM_INSTRUCTION_SPELL = """
Fix any spelling errors in this movie search query. Only correct obvious typos. 
Don't change correctly spelled words
"""

LLM_SYSTEM_INSTRUCTION_REWRITE = """
Rewrite this movie search query to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
"""

LLM_SYSTEM_INSTRUCTION_EXPAND = """
Expand this movie search query with related terms, dont add more than 10 related words.


Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.
Dont repeat meaning too much in thoose terms.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
"""

LLM_SYSTEM_INSTRUCTION_RERANK_INDIVIDUAL = """
Rate how well this movie matches the search query.

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.
"""

LLM_SYSTEM_INSTRUCTION_RERANK_BATCH = """
Rank This list of movies by relevance to the search query
Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

LLM_RRF_SEARCH_EVALUATION = """
Rate how relevant each result is to this query on a 0-3 scale

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

DO NOT give any number out than 0,1,2 or 3.

Return only the scores in the same order you were given. Return a valid JSON list, nothing else.
example
[2,0,3,2,0,1]
"""

LLM_RAG_RESPONSE = """
Answer the question or provide information based on the provided documents. This should be tailored to the app users. the app is a movie streaming service

Provide a comprehensive answer that addresses the query, limit to aproximately 100 words max:
"""

LLM_RAG_SUMMARIZE = """
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

""Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative, try fitting the whole summary of this response in a single paragraph.
- No more than 200 words

Answer:"""

LLM_RAG_QUESTION = """
Answer the following question based on the provided documents.


General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""
