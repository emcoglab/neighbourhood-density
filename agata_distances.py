from os import path
from typing import List

import pandas
from pandas import read_excel

wordlist_path = path.join(path.dirname(__file__), "words for linguistic distance_Agata.xlsx")


def load_word_list() -> List[str]:
    read_excel(wordlist_path)


if __name__ == '__main__':
    print(len(load_word_list()))
