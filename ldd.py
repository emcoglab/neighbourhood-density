import logging

from numpy import array, mean, squeeze, nan

from constants import LDD_WORDS
from ldm.model.base import VectorSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.exceptions import WordNotFoundError
from ldm.utils.logging import print_progress
from ldm.utils.maths import DistanceType


_logger = logging.getLogger(__name__)


def ldds_from_ngram_model(model: NgramModel, wordlist: list[str], neighbourhood_size: int) -> list[float]:
    similarity_matrix = model.underlying_count_model.matrix
    similarity_matrix.eliminate_zeros()  # drop zeros to ensure that min value is non-zero
    # Convert similarities to distances by linearly swapping max and non-zero min values
    max_similarity = similarity_matrix.data.max()
    min_similarity = similarity_matrix.data.min()
    assert min_similarity == abs(min_similarity)

    def similarities_to_distance(similarities):
        return min_similarity + max_similarity - similarities

    ldds: list[float] = []
    nearest_words: list[tuple[str, ...]] = []
    for i, word in enumerate(wordlist, start=1):
        print_progress(i, len(wordlist), bar_length=50, suffix=f" ({i:,}/{LDD_WORDS:,})")

        try:
            try:
                idx = model.underlying_count_model.token_index.token2id[word]
            except KeyError:
                raise WordNotFoundError(word)

            similarities = squeeze(
                array(similarity_matrix[idx, :LDD_WORDS].todense()))  # Only considering most frequent words
            distances = similarities_to_distance(similarities)

            # closest neighbours
            # argsort gives the idxs of the nearest neighbours
            nearest_idxs = distances.argsort()

            # in case a "nearest neighbour" is itself, we look for and remove that idx.
            # but we can speed up the search by first truncating down to the nearest N+2 members
            nearest_idxs = nearest_idxs[:neighbourhood_size + 1]
            nearest_idxs = [i for i in nearest_idxs if not i == idx]
            # need to truncate again in case nothing was removed
            nearest_idxs = nearest_idxs[:neighbourhood_size]
            neighbours = list(model.underlying_count_model.token_index.id2token[i] for i in nearest_idxs)
            ldds.append(mean(distances[nearest_idxs]))
        except WordNotFoundError:
            neighbours = []
            ldds.append(nan)

        nearest_words.append((word, *neighbours))

    return ldds


def ldds_from_vector_model(model: VectorSemanticModel, distance_type: DistanceType, wordlist: list[str],
                           neighbourhood_size: int) -> list[float]:

    ldds: list[float] = []
    nearest_words: list[tuple[str, ...]] = []
    for i, word in enumerate(wordlist, start=1):
        print_progress(i, len(wordlist), bar_length=50, suffix=f" ({i:,}/{LDD_WORDS:,})")

        try:
            neighbours_with_distances = model.nearest_neighbours_with_distances(word, distance_type=distance_type,
                                                                                n=neighbourhood_size,
                                                                                only_consider_most_frequent=LDD_WORDS)
            neighbours = [n for n, d in neighbours_with_distances]
            distances = array([d for n, d in neighbours_with_distances])
            ldds.append(mean(distances))
        except WordNotFoundError:
            neighbours = []
            ldds.append(nan)

        nearest_words.append((word, *neighbours))

    return ldds
