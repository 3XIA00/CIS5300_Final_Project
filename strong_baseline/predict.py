"""Write predictions for a split from a checkpoint saved by strong_baseline.train.

    python -m strong_baseline.predict --checkpoint checkpoints/roberta-base/seed13 --split test \
        --out output/roberta-base/seed13/test.jsonl
    python score.py --gold data/processed/test.jsonl --pred output/roberta-base/seed13/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from strong_baseline.data import label_names, load_label_map, load_split
from strong_baseline.model import UtteranceClassifier
from strong_baseline.train import Encoded, make_collate, pick_device, predict, write_predictions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="folder with model.pt and config.json")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--out", required=True, help="prediction JSONL to write")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    label_map = load_label_map()
    names = label_names(label_map)
    if names != config["labels"]:
        raise SystemExit(f"label order in checkpoint {config['labels']} differs from data/label_map.json {names}")

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = UtteranceClassifier(config["model"], config["num_labels"], dropout=config["dropout"])
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu"))
    model.to(device)

    records = load_split(args.split)
    dataset = Encoded(records, tokenizer, label_map, config["max_length"], config["history"])
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=make_collate(tokenizer.pad_token_id))
    write_predictions(Path(args.out), records, predict(model, loader, device, names, fp16=False))
    print(f"wrote {len(records)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
