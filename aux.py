from os import path
from typing import List, Optional

from pandas import read_excel, DataFrame

from ldm.utils.maths import DistanceType

LDD_N = 20
WORD_RANK_FREQ_THRESHOLD = 60_000

SAVE_DIR = "/Users/caiwingfield/Desktop/"


def load_word_list() -> List[str]:
    wordlist_path = path.join(path.dirname(__file__), "words for linguistic distance_Agata.xlsx")
    with open(wordlist_path, mode="rb") as wordlist_file:
        wordlist_df = read_excel(wordlist_file)
    wordlist = list(wordlist_df['Word'])
    return [w.lower().strip() for w in wordlist]


def save_files(ldds, model, nearest_words, not_found, distance: Optional[DistanceType]):
    if distance is not None:
        filename_prefix = f"{model.name} {distance.name} (top {WORD_RANK_FREQ_THRESHOLD:,} words)"
    else:
        filename_prefix = f"{model.name} (top {WORD_RANK_FREQ_THRESHOLD:,} words)"

    # Save LDD file
    DataFrame(ldds, columns=["Word", f"LDD{LDD_N}"]).to_csv(
        path.join(SAVE_DIR, f"{filename_prefix} LDD{LDD_N}.csv"), index=False)

    # Save neighbours file
    DataFrame(nearest_words, columns=["Word"] + [f"Neighbour {n}" for n in range(1, LDD_N + 1)]).to_csv(
        path.join(SAVE_DIR, f"{filename_prefix} neighbours.csv"), index=False)

    # Save not-found list
    with open(path.join(SAVE_DIR, f"{filename_prefix} not found.txt"), mode="w",
              encoding="utf-8") as not_found_file:
        for w in not_found:
            not_found_file.write(f"{w}\n")
