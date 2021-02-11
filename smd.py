import logging
from os import path
from pathlib import Path
from sys import argv
from typing import Tuple, List

from numpy import array, mean, reshape, squeeze
from pandas import DataFrame
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from constants import SAVE_DIR, SMD_N
from ldm.utils.lists import unzip
from ldm.utils.logging import print_progress
from ldm.utils.maths import DistanceType
from sm.exceptions import WordNotInNormsError
from sm.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)


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


def main():

    the_norms = SensorimotorNormsDistances()
    wordlist = list(the_norms.iter_words())

    df: DataFrame = DataFrame(wordlist, columns=["Word"])

    for distance_type in DistanceType:
        logger.info(f"Computing neighbourhood densities using {distance_type.name} distance")

        smds: List[float] = []
        nearest_words: List[Tuple] = []
        for i, word in enumerate(wordlist, start=1):
            print_progress(i, len(wordlist), bar_length=50)

            neighbours_with_distances = the_norms.nearest_neighbours_with_distances(word, n=SMD_N, distance_type=distance_type)

            neighbours: Tuple[str]
            distances: array
            neighbours, distances = unzip(neighbours_with_distances)

            smds.append(mean(distances))
            nearest_words.append((word, *neighbours))

        df[f"SMD{SMD_N} ({distance_type.name})"] = smds

        # Save neighbours file
        DataFrame(nearest_words, columns=["Word"] + [f"Neighbour {n}" for n in range(1, SMD_N + 1)]).to_csv(
            path.join(SAVE_DIR, f"{distance_type.name} neighbours.csv"), index=False)

    # Save SMD file
    smd_path = Path(SAVE_DIR, f"smd{SMD_N}.csv")
    with smd_path.open(mode="w") as smd_file:
        df.to_csv(smd_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
