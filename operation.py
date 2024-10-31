from pathlib import Path

from pandas import DataFrame

from ldd import ldds_from_ngram_model, ldds_from_vector_model
from ldm.model.base import VectorSemanticModel, DistributionalSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.maths import DistanceType
from smd import smds


def get_words_from_file(wordlist_path: Path) -> list[str]:
    """Get (non-blank) words from lines in a text file."""
    words = []
    with wordlist_path.open("r") as f:
        for line in f:
            line = line.strip().lower()
            if len(line) > 0:
                words.append(line)
    return words


def linguistic_neighbourhood_densities(words: list[str], neighbourhood_size: int,
                                       model: DistributionalSemanticModel, distance: DistanceType | None) -> DataFrame:
    distances: list[float]
    if isinstance(model, NgramModel):
        distances = ldds_from_ngram_model(model, words, neighbourhood_size)
    elif isinstance(model, VectorSemanticModel):
        distances = ldds_from_vector_model(model, distance, words, neighbourhood_size)
    else:
        raise NotImplementedError()

    return DataFrame({
        "Word": words,
        f"LDD{neighbourhood_size}": distances,
    })


def sensorimotor_neighbourhood_densities(words: list[str], distance: DistanceType, neighbourhood_size: int) -> DataFrame:
    distances = smds(words, distance, neighbourhood_size)

    return DataFrame({
        "Word": words,
        f"LDD{neighbourhood_size}": distances,
    })
