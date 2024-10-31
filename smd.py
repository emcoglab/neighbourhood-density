import logging
from os import path
from pathlib import Path
from typing import Tuple, List

from numpy import array, mean, reshape, squeeze, nan
from pandas import DataFrame
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from ldm.utils.lists import unzip
from ldm.utils.logging import print_progress
from ldm.utils.maths import DistanceType
from sm.exceptions import WordNotInNormsError
from sm.sensorimotor_norms import SensorimotorNorms


_logger = logging.getLogger(__name__)


class SensorimotorNormsDistances(SensorimotorNorms):
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


def smds(wordlist: list[str], distance: DistanceType, neighbourhood_size: int) -> list[float]:

    the_norms = SensorimotorNormsDistances()

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
