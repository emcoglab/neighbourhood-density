import logging
from os import path
from pathlib import Path
from sys import argv
from typing import Tuple, List

from numpy import array, mean, squeeze
from pandas import DataFrame

from constants import LDD_WORDS, SAVE_DIR, LDD_N
from ldm.corpus.indexing import FreqDist
from ldm.model.count import CountVectorModel, PPMIModel
from ldm.model.ngram import PPMINgramModel, NgramModel
from ldm.preferences.preferences import Preferences as LDMPreferences
from ldm.utils.logging import print_progress
from ldm.utils.maths import DistanceType

logger = logging.getLogger(__name__)


def ldds_from_ngram_model(model: NgramModel, wordlist):
    similarity_matrix = model.underlying_count_model.matrix
    similarity_matrix.eliminate_zeros()  # drop zeros to ensure that min value is non-zero
    # Convert similarities to distances by linearly swapping max and non-zero min values
    max_similarity = similarity_matrix.data.max()
    min_similarity = similarity_matrix.data.min()
    assert min_similarity == abs(min_similarity)

    def similarities_to_distance(similarities):
        return min_similarity + max_similarity - similarities

    ldds: List[float] = []
    nearest_words: List[Tuple] = []
    for i, word in enumerate(wordlist, start=1):
        print_progress(i, len(wordlist), bar_length=50, suffix=f" ({i:,}/{LDD_WORDS:,})")

        idx = model.underlying_count_model.token_index.token2id[word]

        similarities = squeeze(
            array(similarity_matrix[idx, :LDD_WORDS].todense()))  # Only considering most frequent words
        distances = similarities_to_distance(similarities)

        # closest neighbours
        # argsort gives the idxs of the nearest neighbours
        nearest_idxs = distances.argsort()

        # in case a "nearest neighbour" is itself, we look for and remove that idx.
        # but we can speed up the search by first truncating down to the nearest N+2 members
        nearest_idxs = nearest_idxs[:LDD_N + 1]
        nearest_idxs = [i for i in nearest_idxs if not i == idx]
        # need to truncate again in case nothing was removed
        nearest_idxs = nearest_idxs[:LDD_N]
        neighbours = list(model.underlying_count_model.token_index.id2token[i] for i in nearest_idxs)

        ldds.append(mean(distances[nearest_idxs]))
        nearest_words.append((word, *neighbours))

    # Save neighbours file
    DataFrame(nearest_words, columns=["Word"] + [f"Neighbour {n}" for n in range(1, LDD_N + 1)]).to_csv(
        path.join(SAVE_DIR, f"{model.name} neighbours.csv"), index=False)

    return ldds


def ldds_from_vector_model(model: CountVectorModel, distance_type: DistanceType, wordlist):

    # drop zeros to ensure that min value is non-zero
    ldds = []
    nearest_words = []
    for i, word in enumerate(wordlist, start=1):
        print_progress(i, len(wordlist), bar_length=50, suffix=f" ({i:,}/{LDD_WORDS:,})")

        neighbours_with_distances = model.nearest_neighbours_with_distances(word, distance_type=distance_type,
                                                                            n=LDD_N,
                                                                            only_consider_most_frequent=LDD_WORDS)

        distances = array([d for n, d in neighbours_with_distances])

        ldds.append(mean(distances))

    # Save neighbours file
    DataFrame(nearest_words, columns=["Word"] + [f"Neighbour {n}" for n in range(1, LDD_N + 1)]).to_csv(
        path.join("/Users/caiwingfield/Desktop/", f"{model.name} neighbours.csv"), index=False)

    return ldds


def main():

    corpus = LDMPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)

    wordlist = sorted(freq_dist.most_common_tokens(LDD_WORDS))

    df: DataFrame = DataFrame([
        {"Word": word, f"Frequency ({corpus.name})": freq_dist[word]}
        for word in wordlist
        ])

    model = PPMINgramModel(corpus_meta=corpus, window_radius=5, freq_dist=freq_dist)
    model.train(memory_map=True)

    logger.info(f"Computing neighbourhood densities using {model.name}")
    df[f"LDD{LDD_N} ({model.name})"] = ldds_from_ngram_model(model, wordlist)

    model.untrain()
    model = PPMIModel(corpus, window_radius=5, freq_dist=freq_dist)
    model.train(memory_map=True)

    logger.info(f"Computing neighbourhood densities using {model.name}")
    df[f"LDD{LDD_N} ({model.name})"] = ldds_from_vector_model(model, DistanceType.cosine, wordlist)

    model.untrain()

    # Save LDD file
    ldd_path = Path(SAVE_DIR, f"ldd{LDD_N}.csv")
    with ldd_path.open(mode="w") as ldd_file:
        df.to_csv(ldd_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
