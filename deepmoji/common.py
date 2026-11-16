"""Shared paths and torchMoji loading for the deepmoji package."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCHMOJI_DIR = ROOT / "third_party" / "torchMoji"  # created by deepmoji.setup_torchmoji, git-ignored
FEATURE_DIR = ROOT / "features" / "deepmoji"       # extracted features, git-ignored (regenerated in seconds)
SPLITS = ("train", "dev", "test")
FEATURE_DIM = 2304  # attention-layer output: embedding 256 + 2 x biLSTM 1024 (experiment_plan.md)


def import_torchmoji():
    """Make the patched checkout importable without installing it as a package."""
    if not (TORCHMOJI_DIR / "torchmoji").is_dir():
        sys.exit(f"{TORCHMOJI_DIR} not found: run  python -m deepmoji.setup_torchmoji  first")
    if str(TORCHMOJI_DIR) not in sys.path:
        sys.path.insert(0, str(TORCHMOJI_DIR))


def load_vocabulary():
    import_torchmoji()
    from torchmoji.global_variables import VOCAB_PATH
    with open(VOCAB_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def sentence_tokenizer(fixed_length):
    """torchMoji's own tokenizer. Lower-cases, maps Unicode punctuation (e.g. curly quotes) to ASCII,
    shortens 3+ repeated characters to 2 and looks words up in the 50k DeepMoji vocabulary."""
    import_torchmoji()
    from torchmoji.sentence_tokenizer import SentenceTokenizer
    return SentenceTokenizer(load_vocabulary(), fixed_length)


def feature_encoder(pretrained=True, seed=None):
    """Frozen DeepMoji encoder returning the 2304-dim representation (not the 64 emoji probabilities).

    pretrained=False builds the same architecture with random weights from `seed`
    (the random-feature control, conditions R / CR in experiment_plan.md).
    """
    import_torchmoji()
    import torch
    from torchmoji.global_variables import NB_TOKENS, PRETRAINED_PATH
    from torchmoji.model_def import TorchMoji, torchmoji_feature_encoding

    if pretrained:
        model = torchmoji_feature_encoding(PRETRAINED_PATH)
    else:
        if seed is None:
            raise ValueError("a random encoder needs a seed")
        torch.manual_seed(seed)
        model = TorchMoji(nb_classes=None, nb_tokens=NB_TOKENS, feature_output=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
