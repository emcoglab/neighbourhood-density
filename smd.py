import logging
from os import path
from sys import argv
from typing import Tuple, Optional, List

from numpy import array, mean, reshape, squeeze
from pandas import DataFrame
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix

from ldm.utils.lists import unzip
from ldm.utils.maths import DistanceType
from sm.exceptions import WordNotInNormsError
from sm.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)

SMD_N = 20
SAVE_DIR = "/Users/caiwingfield/Box Sync/LANGBOOT Project/Model/SMD20"


class SensorimotorNormsDistances(SensorimotorNorms):
    """Extension of SensorimotorNorms which can compute distances and neighbourhoods."""

    def _distances_for_word(self, word: str, distance_type: DistanceType) -> array:
        """Vector of distances from the specified word."""
        vector = self.vector_for_word(word)

        if distance_type in [DistanceType.cosine, DistanceType.Euclidean, DistanceType.correlation]:
            return distance_matrix(vector, self.matrix(), metric=distance_type.name)
        elif distance_type == DistanceType.Minkowski3:
            # squeeze to undo the inner reshape
            return squeeze(
                minkowski_distance_matrix(
                    # minkowski_distance_matrix requires a matrix, not a vector
                    reshape(vector, (1, len(vector))),
                    self.matrix(), 3))
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


def save_files(smds, nearest_words, not_found, distance: Optional[DistanceType]):
    # Save SMD file
    DataFrame(smds, columns=["Word", f"SMD{SMD_N}"]).to_csv(
        path.join(SAVE_DIR, f"{distance.name} SMD{SMD_N}.csv"), index=False)

    # Save neighbours file
    DataFrame(nearest_words, columns=["Word"] + [f"Neighbour {n}" for n in range(1, SMD_N + 1)]).to_csv(
        path.join(SAVE_DIR, f"{distance.name} neighbours.csv"), index=False)

    # Save not-found list
    with open(path.join(SAVE_DIR, f"{distance.name} not found.txt"), mode="w",
              encoding="utf-8") as not_found_file:
        for w in not_found:
            not_found_file.write(f"{w}\n")


def main():

    sm = SensorimotorNormsDistances()

    wordlist = list(sm.iter_words())

    distance_type = DistanceType.Minkowski3

    smds = []
    nearest_words = []
    not_found = []
    for word_i, word in enumerate(wordlist):
        try:
            neighbours_with_distances = sm.nearest_neighbours_with_distances(
                word,
                n=SMD_N,
                distance_type=distance_type)
        except WordNotInNormsError:
            not_found.append(word)
            continue

        neighbours: Tuple[str]
        distances: array
        neighbours, distances = unzip(neighbours_with_distances)

        smd = mean(distances)

        smds.append((word, smd))

        nearest_words.append((word, *neighbours))

        if (word_i % 100 == 0) and (word_i > 0):
            logger.info(f"Done {word_i:,}/{len(wordlist):,} ({100 * word_i / len(wordlist):.2f}%)")

    save_files(smds, nearest_words, not_found, distance_type)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
