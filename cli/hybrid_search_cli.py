import argparse

from lib.hybrid_search import hybrid_search, normalize_scores, rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command",help="Available commands")
    
    normalize_parser = subparser.add_parser("normalize", help="Normalize BM25 scores for cosine comparison")
    normalize_parser.add_argument("scores" ,type=float, nargs='*', help="Scores to normalize")

    weighted_search_command = subparser.add_parser("weighted-search",help="Search with BM25 Keyword and Semantic Search")
    weighted_search_command.add_argument("query",type=str,help="query to search")
    weighted_search_command.add_argument("--alpha",type=float,help="alpha for weights")
    weighted_search_command.add_argument("--limit", type=int,help="search limit")

    rff_search_command = subparser.add_parser("rrf-search",help="Reciprocal Rank Fusion")
    rff_search_command.add_argument("query",type=str,help="query to search")
    rff_search_command.add_argument("-k",type=int, default=60,help="k for RFF")
    rff_search_command.add_argument("--limit",type=int, default=5,help="search limit")
    rff_search_command.add_argument("--enhance",type=str, choices=["spell","rewrite","expand"], help="Query enhancment method")
    rff_search_command.add_argument("--rerank-method",type=str, choices=["individual","batch","cross_encoder"], help="Rerank method")
    rff_search_command.add_argument("--evaluate", action="store_true", help="LLM evaluation")

    


    args = parser.parse_args()

    match args.command:
        
        case "normalize":
            normalize_scores(args.scores)

        case "weighted-search":
            hybrid_search(args.query,args.alpha,args.limit)

        case "rrf-search":
            rrf_search(args.query,args.k,args.limit,args.enhance,args.rerank_method,args.evaluate)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
