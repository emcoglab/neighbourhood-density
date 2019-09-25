import logging
from sys import argv

from numpy import mean, array, squeeze

from aux import LDD_N, load_word_list, WORD_RANK_FREQ_THRESHOLD, save_files
from ldm.corpus.indexing import FreqDist
from ldm.model.ngram import PPMINgramModel
from ldm.preferences.preferences import Preferences as LDMPreferences

logger = logging.getLogger(__name__)


def main():
    corpus = LDMPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)

    models = [
        PPMINgramModel(corpus_meta=corpus, window_radius=5, freq_dist=freq_dist)
    ]

    for model in models:
        model.train(memory_map=True)
        ldd_from_model(model)
        model.untrain()


def ldd_from_model(model):
    wordlist = load_word_list()
    similarity_matrix = model.underlying_count_model.matrix
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
            idx = model.underlying_count_model.token_index.token2id[word]
        except KeyError:
            not_found.append(word)
            continue

        ocmf = WORD_RANK_FREQ_THRESHOLD
        if ocmf is not None:
            similarities = squeeze(array(similarity_matrix[idx, :WORD_RANK_FREQ_THRESHOLD].todense()))
        else:
            similarities = squeeze(array(similarity_matrix[idx, :].todense()))

        # Convert similarities to distances by subtracting from the max value
        distances = max_similarity - similarities

        # closest neighbours
        # argsort gives the idxs of the nearest neighbours
        nearest_idxs = distances.argsort()

        # in case a "nearest neighbour" is itself, we look for and remove that idx.
        # but we can speed up the search by first truncating down to the nearest N+2 members
        nearest_idxs = nearest_idxs[:LDD_N + 1]
        nearest_idxs = [i for i in nearest_idxs if not i == idx]
        # need to truncate again in case nothing was removed
        nearest_idxs = nearest_idxs[:LDD_N]

        nearest_words.append((word, *(model.underlying_count_model.token_index.id2token[i] for i in nearest_idxs)))

        ldd = mean(distances[nearest_idxs])

        ldds.append((word, ldd))

        if word_count % 100 == 0:
            logger.info(f"Done {word_count:,}/{len(wordlist):,} ({100 * word_count / len(wordlist):.2f}%)")

    save_files(ldds, model, nearest_words, not_found)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
