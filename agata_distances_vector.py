import logging
from os import path
from sys import argv

from numpy import mean, array
from pandas import DataFrame

from aux import LDD_N, load_word_list, ONLY_CONSIDER_MOST_FREQUENT
from ldm.corpus.indexing import FreqDist
from ldm.model.count import LogCoOccurrenceCountModel, PPMIModel, CountVectorModel
from ldm.preferences.preferences import Preferences as LDMPreferences
from ldm.utils.exceptions import WordNotFoundError
from ldm.utils.maths import DistanceType

logger = logging.getLogger(__name__)


def main():
    corpus = LDMPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)

    models = [
        LogCoOccurrenceCountModel(corpus_meta=corpus, window_radius=5, freq_dist=freq_dist),
        PPMIModel(corpus_meta=corpus, window_radius=5, freq_dist=freq_dist)
    ]

    distance = DistanceType.cosine

    for model in models:
        model.train(memory_map=True)
        ldd_from_model(model, distance)
        model.untrain()


def ldd_from_model(model: CountVectorModel, distance_type: DistanceType):
    wordlist = load_word_list()

    similarity_matrix = model.matrix
    # drop zeros to ensure that min value is non-zero
    similarity_matrix.eliminate_zeros()
    max_similarity = similarity_matrix.data.max()
    # Use the absolute value in case the minimum is negative (e.g. with PMI).
    min_similarity = abs(similarity_matrix.data.min())
    ldds = []
    nearest_words = []
    not_found = []
    for word_count, word in enumerate(wordlist, start=1):
        try:
            neighbours_with_distances = model.nearest_neighbours_with_distances(word, distance_type=distance_type, n=LDD_N, only_consider_most_frequent=ONLY_CONSIDER_MOST_FREQUENT)
        except WordNotFoundError:
            not_found.append(word)
            continue

        distances = array([d for n, d in neighbours_with_distances])

        ldd = mean(distances)

        ldds.append((word, ldd))

        if word_count % 100 == 0:
            logger.info(f"Done {word_count:,}/{len(wordlist):,} ({100 * word_count / len(wordlist):.2f}%)")
    # Save LDD file
    DataFrame(ldds, columns=["Word", f"LDD{LDD_N}"]).to_csv(
        path.join("/Users/caiwingfield/Desktop/", f"{model.name} LDD{LDD_N}.csv"), index=False)
    # Save neighbours file
    DataFrame(nearest_words, columns=["Word"] + [f"Neighbour {n}" for n in range(1, LDD_N + 1)]).to_csv(
        path.join("/Users/caiwingfield/Desktop/", f"{model.name} neighbours.csv"), index=False)
    # Save not-found list
    with open(path.join("/Users/caiwingfield/Desktop/", f"{model.name} not found.txt"), mode="w",
              encoding="utf-8") as not_found_file:
        for w in not_found:
            not_found_file.write(f"{w}\n")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
