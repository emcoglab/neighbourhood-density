from os import path
from typing import List

from pandas import read_excel

LDD_N = 20
ONLY_CONSIDER_MOST_FREQUENT = 60_000

def load_word_list() -> List[str]:
    wordlist_path = path.join(path.dirname(__file__), "words for linguistic distance_Agata.xlsx")
    with open(wordlist_path, mode="rb") as wordlist_file:
        wordlist_df = read_excel(wordlist_file)
    wordlist = list(wordlist_df['Word'])
    return [w.lower().strip() for w in wordlist]