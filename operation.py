from pathlib import Path
from typing import List, Tuple

from numpy import array, reshape, squeeze, mean, nan
from pandas import DataFrame
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from constants import LDD_WORDS
from ldm.model.base import VectorSemanticModel, DistributionalSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.exceptions import WordNotFoundError
from ldm.utils.lists import unzip
from ldm.utils.logging import print_progress
from ldm.utils.maths import DistanceType
from sm.exceptions import WordNotInNormsError
from sm.sensorimotor_norms import SensorimotorNorms


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
        distances = linguistic_densities_from_ngram_model(model, words, neighbourhood_size)
    elif isinstance(model, VectorSemanticModel):
        distances = linguistic_densities_from_vector_model(model, distance, words, neighbourhood_size)
    else:
        raise NotImplementedError()

    return DataFrame({
        "Word": words,
        f"LDD{neighbourhood_size}": distances,
    })


def sensorimotor_neighbourhood_densities(words: list[str], distance: DistanceType, neighbourhood_size: int) -> DataFrame:
    distances = sensorimotor_densities_from_norms(words, distance, neighbourhood_size)

    return DataFrame({
        "Word": words,
        f"LDD{neighbourhood_size}": distances,
    })


class _SensorimotorNormsDistances(SensorimotorNorms):
    """Extension of SensorimotorNorms which can compute distances and neighbourhoods."""

    def _distances_for_word(self, word: str, distance_type: DistanceType) -> array:
        """Vector of distances from the specified word."""
        vector = self.vector_for_word(word)
        # pairwise distances functions require a matrix, not a vector
        vector = reshape(vector, (1, len(vector)))

        if distance_type in [DistanceType.cosine, DistanceType.Euclidean, DistanceType.correlation]:
            # squeeze to undo the inner reshape
            return squeeze(
                distance_matrix(vector, self.matrix(), metric=distance_type.name))
        elif distance_type == DistanceType.Minkowski3:
            # squeeze to undo the inner reshape
            return squeeze(
                minkowski_distance_matrix(vector, self.matrix(), 3))
        else:
            raise NotImplementedError()

    def nearest_neighbours_with_distances(self,
                                          word: str,
                                          distance_type: DistanceType,
                                          n: int,
                                          ) -> List[Tuple[str, float]]:
        """
        :param word:
        :param distance_type:
        :param n:
        :return:
        :raises: WordNotInNormsError
        """
        wordlist = list(self.iter_words())

        if not self.has_word(word):
            raise WordNotInNormsError(f"The word {word!r} was not found.")

        distances = self._distances_for_word(word, distance_type)

        # Get the indices of the largest and smallest distances
        nearest_idxs = distances.argsort()

        # in case a "nearest neighbour" is itself, we look for and remove that idx.
        # but we can speed up the search by first truncating down to the nearest N+2 members
        nearest_idxs = nearest_idxs[:n + 1]
        nearest_idxs = [i for i in nearest_idxs if not wordlist[i] == word]
        # need to truncate again in case nothing was removed
        nearest_idxs = nearest_idxs[:n]

        nearest_neighbours = [(wordlist[i], distances[i]) for i in nearest_idxs]

        return nearest_neighbours


def sensorimotor_densities_from_norms(wordlist: list[str], distance: DistanceType, neighbourhood_size: int) -> list[float]:

    the_norms = _SensorimotorNormsDistances()

    smds: list[float] = []
    nearest_words: list[tuple[str, ...]] = []
    for i, word in enumerate(wordlist, start=1):
        print_progress(i, len(wordlist), bar_length=50)

        try:
            neighbours_with_distances = the_norms.nearest_neighbours_with_distances(word, n=neighbourhood_size,
                                                                                    distance_type=distance)

            neighbours: tuple[str, ...]
            distances: array
            neighbours, distances = unzip(neighbours_with_distances)

            smds.append(mean(distances))

        except WordNotInNormsError:
            neighbours = []
            smds.append(nan)

        nearest_words.append((word, *neighbours))

    return smds


def linguistic_densities_from_ngram_model(model: NgramModel, wordlist: list[str], neighbourhood_size: int) -> list[float]:
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


def linguistic_densities_from_vector_model(model: VectorSemanticModel, distance_type: DistanceType, wordlist: list[str],
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
