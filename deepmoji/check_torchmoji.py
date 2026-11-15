"""Check that the patched torchMoji behaves like the original and is usable as a frozen encoder.

    python -m deepmoji.check_torchmoji

1. Reference outputs: reproduces the two numeric checks in torchMoji's tests/test_finetuning.py
   (top-5 emojis for seven sentences, and the mean of the first five feature dimensions).
2. Shape (n, 2304), finite values, identical output on a second pass (evaluation mode, no dropout).
3. Frozen: no parameter requires a gradient.
4. Random-weight control: same shape; the same seed gives identical features, different seeds differ.
Exits with status 1 if any check fails.
"""

import sys

import numpy as np
import torch

from deepmoji.common import FEATURE_DIM, feature_encoder, import_torchmoji, sentence_tokenizer

SENTENCES = ["I love mom's cooking", "I love how you never reply back..", "I love cruising with my homies",
             "I love messing with yo mind!!", "I love you and now you're just gone..", "This is shit",
             "This is the shit"]
# From torchMoji tests/test_finetuning.py (test_score_emoji, test_encode_texts).
EXPECTED_TOP5 = [[36, 4, 8, 16, 47], [1, 19, 55, 25, 46], [31, 6, 30, 15, 13], [54, 44, 9, 50, 49],
                 [46, 5, 27, 35, 34], [55, 32, 27, 1, 37], [48, 11, 6, 31, 9]]
EXPECTED_MEAN5 = [-0.023, 0.021, -0.037, -0.001, -0.005]

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def main():
    import_torchmoji()
    from torchmoji.global_variables import PRETRAINED_PATH
    from torchmoji.model_def import torchmoji_emojis

    tokens, _, _ = sentence_tokenizer(30).tokenize_sentences(SENTENCES)
    with torch.no_grad():
        emoji_model = torchmoji_emojis(PRETRAINED_PATH).eval()
        probs = emoji_model(tokens)
    top5 = [list(map(int, np.argsort(p)[-5:][::-1])) for p in probs]
    check("emoji top-5 matches torchMoji reference", top5 == EXPECTED_TOP5)

    encoder = feature_encoder()
    with torch.no_grad():
        features = encoder(tokens)
        again = encoder(tokens)
    mean5 = np.around(features.mean(0)[:5], 3)
    check("feature mean matches torchMoji reference", bool(np.allclose(mean5, EXPECTED_MEAN5)), str(mean5))
    check("shape", features.shape == (len(SENTENCES), FEATURE_DIM), str(features.shape))
    check("finite values", bool(np.isfinite(features).all()))
    check("deterministic (second pass identical)", bool(np.array_equal(features, again)))
    check("frozen (no parameter requires grad)", not any(p.requires_grad for p in encoder.parameters()))

    with torch.no_grad():
        r13a = feature_encoder(pretrained=False, seed=13)(tokens)
        r13b = feature_encoder(pretrained=False, seed=13)(tokens)
        r42 = feature_encoder(pretrained=False, seed=42)(tokens)
    check("random control shape", r13a.shape == (len(SENTENCES), FEATURE_DIM), str(r13a.shape))
    check("random control: same seed gives identical features", bool(np.array_equal(r13a, r13b)))
    check("random control: different seeds differ", not np.allclose(r13a, r42))

    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")


if __name__ == "__main__":
    main()
