#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_query_text, embed_text, verify_embeddings, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command",help="Available Commands") 

    subparsers.add_parser("verify",help="Verify local model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate Text embeddings")
    embed_text_parser.add_argument("text",type=str,help="text for embedding")

    subparsers.add_parser("verify_embeddings",help="Verify embeddings")
    
    embed_query = subparsers.add_parser("embedquery", help="Embed the search query")
    embed_query.add_argument("query", type=str,help="Search Query")


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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
