import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT, print_rag_results
from lib.rag import question_command, rag_command, summarize

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAF (search + generate answer)"
    )

    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Sumarize search results"
    )

    summarize_parser.add_argument("query", type=str, help="Query to search and summarize")

    question = subparsers.add_parser(
        "question", help="Provide citations in summary search"
    )

    question.add_argument("question",type=str, help="Query to search and summarize")
    question.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit")


    args = parser.parse_args()

    match args.command:
        case "rag":
            results, rag_response = rag_command(args.query)
            print_rag_results(results,rag_response,"RAG Response")

        case "summarize":
            results, llm_summary = summarize(args.query)
            print_rag_results(results,llm_summary,"LLM Summary")

        case "question":
            results, response = question_command(args.question,args.limit)
            print_rag_results(results,response,"Answer")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
