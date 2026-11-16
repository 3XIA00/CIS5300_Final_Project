"""Extract frozen DeepMoji features for every MELD utterance and report unknown-word rates.

    python -m deepmoji.extract_features                      # pretrained DeepMoji
    python -m deepmoji.extract_features --random-seed 13     # random-weight control for one seed

Writes features/deepmoji/<name>/{train,dev,test}.npy (float32, n x 2304) and
{train,dev,test}_ids.json (row order), where <name> is "pretrained" or "random-seed<N>".
DeepMoji sees the utterance text only, without the speaker prefix RoBERTa gets (experiment_plan.md).
"""

import argparse
import json
import re
import time
from collections import Counter

import numpy as np
import torch

from deepmoji.common import FEATURE_DIM, FEATURE_DIR, ROOT, SPLITS, feature_encoder, read_jsonl, sentence_tokenizer

# Longer than any MELD utterance after DeepMoji tokenisation (max 87, train), so nothing is truncated.
FIXED_LENGTH = 200
UNKNOWN = 1  # CUSTOM_UNKNOWN in the DeepMoji vocabulary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--random-seed", type=int, help="use randomly initialised weights from this seed")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output-dir", help="default: features/deepmoji/<pretrained|random-seed<N>>")
    return p.parse_args(argv)


def unknown_stats(tokenizer, texts):
    n_tokens = n_unknown = n_utterances = 0
    words = Counter()
    for text in texts:
        ws = tokenizer.wordgen.get_words(text)
        ids = tokenizer.find_tokens(ws)
        k = sum(1 for t in ids if t == UNKNOWN)
        n_tokens, n_unknown, n_utterances = n_tokens + len(ids), n_unknown + k, n_utterances + (k > 0)
        words.update(w for w, t in zip(ws, ids) if t == UNKNOWN)
    return {"tokens": n_tokens, "unknown_tokens": n_unknown,
            "unknown_token_pct": round(100 * n_unknown / n_tokens, 2),
            "utterances_with_unknown_pct": round(100 * n_utterances / len(texts), 1),
            "top_unknown": words.most_common(15)}


def main(argv=None):
    args = parse_args(argv)
    name = "pretrained" if args.random_seed is None else f"random-seed{args.random_seed}"
    out = FEATURE_DIR / name if args.output_dir is None else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = sentence_tokenizer(FIXED_LENGTH)
    encoder = feature_encoder(pretrained=args.random_seed is None, seed=args.random_seed)

    all_texts = []
    for split in SPLITS:
        records = read_jsonl(ROOT / "data" / "processed" / f"{split}.jsonl")
        texts = [r["text"] for r in records]
        all_texts += texts
        tokens, _, _ = tokenizer.tokenize_sentences(texts)
        lengths = (tokens != 0).sum(1)
        if lengths.min() == 0:
            raise SystemExit(f"{split}: an utterance has no DeepMoji tokens")
        if lengths.max() >= FIXED_LENGTH:
            raise SystemExit(f"{split}: an utterance reaches the {FIXED_LENGTH}-token limit and may be truncated")
        started = time.time()
        with torch.no_grad():
            features = np.concatenate([encoder(tokens[i:i + args.batch_size])
                                       for i in range(0, len(tokens), args.batch_size)]).astype(np.float32)
        assert features.shape == (len(records), FEATURE_DIM) and np.isfinite(features).all()
        np.save(out / f"{split}.npy", features)
        with open(out / f"{split}_ids.json", "w", encoding="utf-8") as handle:
            json.dump([r["id"] for r in records], handle)
        print(f"{split}: {features.shape}, longest {lengths.max()} DeepMoji tokens, "
              f"{time.time() - started:.1f}s -> {out / (split + '.npy')}")

    if args.random_seed is None:
        straight = [t.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
                    for t in all_texts]
        hyphen_split = [re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", t) for t in all_texts]
        stats = {"raw": unknown_stats(tokenizer, all_texts),
                 "curly_quotes_straightened": unknown_stats(tokenizer, straight),
                 "hyphens_between_letters_split": unknown_stats(tokenizer, hyphen_split)}
        for variant, s in stats.items():
            print(f"unknown words [{variant}]: {s['unknown_token_pct']}% of {s['tokens']} tokens, "
                  f"{s['utterances_with_unknown_pct']}% of utterances; top {s['top_unknown'][:8]}")
        with open(out / "unknown_words.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
