#!/usr/bin/env python3

import argparse

from lib.semantic_search import  embed_chunks_command, embed_query_text, embed_text, semantic_chunk_text_command, semantic_search_chunked_command, standard_chunk_text_command, verify_embeddings, verify_model,semantic_search_command

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command",help="Available Commands") 

    subparsers.add_parser("verify",help="Verify local model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate Text embeddings")
    embed_text_parser.add_argument("text",type=str,help="text for embedding")

    subparsers.add_parser("verify_embeddings",help="Verify embeddings")
    
    embed_query = subparsers.add_parser("embedquery", help="Embed the search query")
    embed_query.add_argument("query", type=str,help="Search Query")

    semantic_chunk_search = subparsers.add_parser("search", help="Search command")
    semantic_chunk_search.add_argument("query",type=str,help="Search term(s)")
    semantic_chunk_search.add_argument("--limit",type=int, help="Number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text for embedding")
    chunk_parser.add_argument("text",type=str,  help="Text to chunk")
    chunk_parser.add_argument("--chunk-size",type=int, default=200, help="Chunk size")
    chunk_parser.add_argument("--overlap",type=int, default=2, help="Chunk overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk",help="Semantic Chunking")
    semantic_chunk_parser.add_argument("text",type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size",type=int, default=4, help="Max chunk size")
    semantic_chunk_parser.add_argument("--overlap",type=int, default=0, help="Overlap")

    subparsers.add_parser("embed_chunks",help="Embed chunks for semantic search")

    semantic_chunk_search = subparsers.add_parser("search_chunked", help="Search command")
    semantic_chunk_search.add_argument("query",type=str,help="Search term(s)")
    semantic_chunk_search.add_argument("--limit",type=int, help="Number of results")


    args = parser.parse_args()

    match args.command:

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            results = semantic_search_command(args.query,args.limit)
            for res in results:
                print(f"{res["id"]}.- {res["title"]} ({res["score"]})\n{res["document"][:50]}...")
                print(f"---------------------------------------------------------------")

        case "chunk":
            chunks = standard_chunk_text_command(args.text,args.chunk_size,args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks,1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            chunks = semantic_chunk_text_command(args.text,args.max_chunk_size,args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks,1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            embed_chunks_command()
            
        case "search_chunked":
            semantic_search_chunked_command(args.query,args.limit)


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
