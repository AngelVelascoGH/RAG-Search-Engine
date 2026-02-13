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
