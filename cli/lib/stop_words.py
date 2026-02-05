import pathlib
root = pathlib.Path.cwd()
movies_dataset = root / "data" / "stopwords.txt" 

def get_stop_words() -> list[str]:
    with open(movies_dataset) as file:
        text = file.read()
        words = text.splitlines()
        return words




