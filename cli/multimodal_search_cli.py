import argparse

from lib.multimodal_search import image_search_command

def main():
    parser = argparse.ArgumentParser(description="Image search (Multimodal)")
    subparser = parser.add_subparsers(dest="command",help="Available Commands")
    
    image_search_parser = subparser.add_parser("image_search", help="Image search")
    image_search_parser.add_argument("path",type=str,help="Image path")

    args = parser.parse_args()

    match args.command:
        case "image_search":
            image_search_command(args.path)



if __name__ == "__main__":
    main()
