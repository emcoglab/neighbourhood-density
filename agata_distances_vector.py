import logging
from sys import argv
from typing import Tuple

from numpy import mean, array

from aux import LDD_N, load_word_list, WORD_RANK_FREQ_THRESHOLD, save_files
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

    ldds = []
    nearest_words = []
    not_found = []
    for word_count, word in enumerate(wordlist, start=1):
        try:
            neighbours_with_distances = model.nearest_neighbours_with_distances(
                word,
                n=LDD_N,
                distance_type=distance_type,
                only_consider_most_frequent=WORD_RANK_FREQ_THRESHOLD)
        except WordNotFoundError:
            not_found.append(word)
            continue

        neighbours: Tuple[str] = tuple(n for n, d in neighbours_with_distances)
        distances: array = array([d for n, d in neighbours_with_distances])

        ldd = mean(distances)

        ldds.append((word, ldd))

        nearest_words.append((word, *neighbours))

        if word_count % 100 == 0:
            logger.info(f"Done {word_count:,}/{len(wordlist):,} ({100 * word_count / len(wordlist):.2f}%)")

    save_files(ldds, model, nearest_words, not_found, distance_type)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
