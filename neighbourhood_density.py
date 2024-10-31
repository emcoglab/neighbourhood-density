import argparse
import logging
from enum import StrEnum
from pathlib import Path

from pandas import DataFrame

from constants import NBHD_SIZE_DEFAULT
from ldm.corpus.corpus import CorpusMetadata
from ldm.corpus.indexing import FreqDist
from ldm.preferences.config import Config as LDMConfig
from ldm.utils.maths import DistanceType
from operation import get_words_from_file, linguistic_neighbourhood_densities, sensorimotor_neighbourhood_densities

# Suppress logging
logger = logging.getLogger('my-logger')
logger.propagate = False

# shortname → dirname
_corpora = {
    "bnc": "BNC",
    "subtitles": "BBC",
    "ukwac": "UKWAC",
}

_ngram_models = [
    "log-ngram",
    "probability-ratio-ngram",
    "ppmi-ngram",
]
_count_models = [
    "log-cooccurrence",
    "conditional-probability",
    "probability-ratio",
    "ppmi",
]
_predict_models = [
    "skip-gram",
    "cbow",
]
_models = _ngram_models + _count_models + _predict_models

_embedding_sizes = [50, 100, 200, 300, 500]
_window_radii = [1, 3, 5, 10]

_readme_path = Path(Path(__file__).parent, "README.md")
_config_path = Path(Path(__file__).parent, "config.yaml")


class Space(StrEnum):
    """The space in which to compute neighbourhood density."""
    lingusitic = "linguistic"
    sensorimotor = "sensorimotor"

    @property
    def shorthand(self) -> str:
        if self == Space.lingusitic:
            return "ldd"
        if self == Space.sensorimotor:
            return "smd"
        raise NotImplementedError()


class WordMode(StrEnum):
    """The mode of operation wrt where to read words from."""
    # One word from CLI
    word_from_cli = "word from CLI"
    # Lis tof words from file
    words_from_file = "words from file"


def build_argparser():
    argparser = argparse.ArgumentParser(description="Compute neighbourhood densities.")

    # Add mode parsers
    mode_subparsers = argparser.add_subparsers(dest="space")
    mode_subparsers.required = True

    mode_ldd_parser = mode_subparsers.add_parser(
        Space.lingusitic.shorthand,
        help="Compute linguistic-distributional neighbourhood density")
    mode_smd_parser = mode_subparsers.add_parser(
        Space.sensorimotor.shorthand,
        help="Compute sensorimotor neighbourhood density")

    for mode_subparser in [mode_ldd_parser, mode_smd_parser]:
        # Neighbourhood specific
        mode_subparser.add_argument("--neighbourhood-size", required=False, type=int,
                                    default=NBHD_SIZE_DEFAULT, help="The size of the neighbourhood.")
        # Input specific
        wordmode_group = mode_subparser.add_mutually_exclusive_group()
        wordmode_group.add_argument("--word", required=False, type=str,
                                    help="The word to look up.")
        wordmode_group.add_argument("--words-from-file", required=False, type=Path,
                                    dest="words_from_file", metavar="PATH",
                                    help="The word to look up or compare.")
        # Output specific
        mode_subparser.add_argument("--output-file", required=False, type=Path,
                                    dest="output_file", metavar="PATH",
                                    help="Write the output to this file.  Will overwrite existing files.")
    # Model specific (linguistic only)
    mode_ldd_parser.add_argument("--corpus", required=True, type=str, choices=_corpora.keys(),
                                 help="The name of the corpus.")
    mode_ldd_parser.add_argument("--model", required=True, nargs="+", type=str,
                                 choices=_models,
                                 dest="model", metavar=("MODEL", "EMBEDDING"),
                                 help="The model specification to use.")
    mode_ldd_parser.add_argument("--radius", required=True, type=int, choices=_window_radii,
                                 dest="window_radius",
                                 help="The window radius to use.")
    # Both models use distances
    mode_ldd_parser.add_argument("--distance", required=False, type=str,
                                 choices=[dt.name for dt in DistanceType],
                                 help="The distance type to use.")
    # But it's required with sensorimotor
    mode_smd_parser.add_argument("--distance", required=True, type=str,
                                 choices=[dt.name for dt in DistanceType],
                                 help="The distance type to use.")

    return argparser


def get_model_from_parameters(model_type: str, window_radius, embedding_size, corpus, freq_dist):
    if model_type is None:
        return None
    # Don't care about difference between underscores and hyphens
    model_type = model_type.lower().replace("_", "-")
    # N-gram models
    if model_type == "log-ngram":
        from ldm.model.ngram import LogNgramModel
        return LogNgramModel(corpus, window_radius, freq_dist)
    if model_type == "probability-ratio-ngram":
        from ldm.model.ngram import ProbabilityRatioNgramModel
        return ProbabilityRatioNgramModel(corpus, window_radius, freq_dist)
    if model_type == "pmi-ngram":
        from ldm.model.ngram import PMINgramModel
        return PMINgramModel(corpus, window_radius, freq_dist)
    if model_type == "ppmi-ngram":
        from ldm.model.ngram import PPMINgramModel
        return PPMINgramModel(corpus, window_radius, freq_dist)
    # Count vector models:
    if model_type == "log-cooccurrence":
        from ldm.model.count import LogCoOccurrenceCountModel
        return LogCoOccurrenceCountModel(corpus, window_radius, freq_dist)
    if model_type == "conditional-probability":
        from ldm.model.count import ConditionalProbabilityModel
        return ConditionalProbabilityModel(corpus, window_radius, freq_dist)
    if model_type == "probability-ratio":
        from ldm.model.count import ProbabilityRatioModel
        return ProbabilityRatioModel(corpus, window_radius, freq_dist)
    if model_type == "pmi":
        from ldm.model.count import PMIModel
        return PMIModel(corpus, window_radius, freq_dist)
    if model_type == "ppmi":
        from ldm.model.count import PPMIModel
        return PPMIModel(corpus, window_radius, freq_dist)
    # Predict vector models:
    if model_type == "skip-gram":
        from ldm.model.predict import SkipGramModel
        return SkipGramModel(corpus, window_radius, embedding_size)
    if model_type == "cbow":
        from ldm.model.predict import CbowModel
        return CbowModel(corpus, window_radius, embedding_size)

    raise NotImplementedError()


def main(ldm_config: LDMConfig):

    argparser = build_argparser()

    args = argparser.parse_args()

    def option_used(option_name):
        if option_name in vars(args):
            if vars(args)[option_name]:
                return True
            else:
                return False
        else:
            return False

    # Get space

    space: Space
    if args.space == Space.lingusitic.shorthand:
        space = Space.lingusitic
    elif args.space == Space.sensorimotor.shorthand:
        space = Space.sensorimotor
    else:
        raise NotImplementedError()

    # Get wordmode
    wordmode: WordMode
    if option_used("word"):
        wordmode = WordMode.word_from_cli
    elif option_used("word_from_file"):
        wordmode = WordMode.words_from_file
    else:
        raise NotImplementedError()

    # Validate model params
    if option_used("model"):
        # For predict models, embedding size is required
        if args.model[0].lower() in _predict_models:
            if len(args.model) == 1:
                argparser.error("Please specify embedding size when using predict models")
            elif int(args.model[1]) not in _embedding_sizes:
                argparser.error(f"Invalid embedding size {args.model[1]}, "
                                f"Please select an embedding size from the list {_embedding_sizes}")

        # For count and ngram models, embedding size is forbidden
        else:
            if len(args.model) > 1:
                argparser.error("Embedding size invalid for count and n-gram models")

    # get model spec
    model_type: str | None
    embedding_size: int | None
    if not option_used("model"):
        model_type = None
        embedding_size = None
    elif len(args.model) == 1:
        model_type = args.model[0]
        embedding_size = None
    elif len(args.model) == 2:
        model_type = args.model[0]
        embedding_size = int(args.model[1])
    else:
        raise NotImplementedError()
    radius: int | None = int(args.window_radius) if "window_radius" in vars(args) else None

    distance: DistanceType | None = None
    for dt in DistanceType:
        if args.distance.lower() == dt.name.lower():
            distance = dt
            break

    # Get corpus and freqdist
    corpus_name = args.corpus
    corpus: CorpusMetadata = CorpusMetadata(
        name=_corpora[corpus_name],
        path=ldm_config.value_by_key_path("corpora", corpus_name, "path"),
        freq_dist_path=ldm_config.value_by_key_path("corpora", corpus_name, "index"))
    freq_dist: FreqDist = FreqDist.load(corpus.freq_dist_path)

    neighbourhood_size: int = int(args.neighbourhood_size)

    # Build model
    model = get_model_from_parameters(model_type, radius, embedding_size, corpus, freq_dist)

    # Get output file
    output_file: Path | None = args.output_file
    if output_file is not None and output_file.exists():
        raise FileExistsError(output_file)

    # Get word(s)
    words: list[str]
    if wordmode == WordMode.word_from_cli:
        words = [args.word.lower()]
    elif wordmode == WordMode.words_from_file:
        words = get_words_from_file(args.words_from_file)
    else:
        raise NotImplementedError()

    # Run appropriate function based on space
    densities: DataFrame
    if space == Space.lingusitic:
        densities = linguistic_neighbourhood_densities(words, neighbourhood_size,
                                                       model, distance)
    elif space == Space.sensorimotor:
        densities = sensorimotor_neighbourhood_densities(words, neighbourhood_size)
    else:
        raise NotImplementedError()

    if output_file is not None:
        densities.to_csv(output_file)
    else:
        for _, word, density in densities.itertuples():
            print(f"{word}: {density}")


if __name__ == '__main__':
    with LDMConfig(use_config_overrides_from_file=str(_config_path)) as config:
        main(config)
