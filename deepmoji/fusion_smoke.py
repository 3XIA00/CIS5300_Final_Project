"""Smoke test: RoBERTa-base <s> state concatenated with frozen DeepMoji features runs end to end.

    python -m deepmoji.fusion_smoke

Uses the strong baseline's own Encoded / make_collate / UtteranceClassifier(aux_dim=2304) / evaluate,
without changing them. First 64 utterances of train and dev, one epoch, batch 16, AdamW lr 2e-5.
Checks shapes, that gradients reach both parts of the classifier, prediction and scoring, and the
shape guard in model.py. The scores on 64 utterances are not meaningful; this is not an experiment.
Also prints two numbers worth watching in Milestone 3: standardised outliers and the gradient ratio.
"""

import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from deepmoji.common import FEATURE_DIM, FEATURE_DIR, ROOT

sys.path.insert(0, str(ROOT))
from strong_baseline.data import label_names, load_label_map, load_split  # noqa: E402
from strong_baseline.model import UtteranceClassifier  # noqa: E402
from strong_baseline.train import Encoded, evaluate, make_collate  # noqa: E402

LIMIT, MODEL, SEED = 64, "roberta-base", 13


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map()
    names = label_names(label_map)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    collate = make_collate(tokenizer.pad_token_id)

    feature_dir = FEATURE_DIR / "pretrained"
    raw = {s: np.load(feature_dir / f"{s}.npy") for s in ("train", "dev", "test")}
    row = {s: {i: k for k, i in enumerate(json.load(open(feature_dir / f"{s}_ids.json", encoding="utf-8")))}
           for s in raw}
    mean, std = raw["train"].mean(0), raw["train"].std(0)  # full train split only (experiment_plan.md)
    print(f"DeepMoji dims: zero-variance on train {int((std == 0).sum())}, smallest std {std.min():.1e}")
    for s in ("dev", "test"):
        z = np.abs((raw[s] - mean) / std)
        print(f"  {s}: largest |z| after standardising {z.max():.1f}, share of values with |z| > 10: "
              f"{100 * (z > 10).mean():.3f}%")

    def split_data(name):
        records = load_split(name)[:LIMIT]
        aux = (raw[name][[row[name][r["id"]] for r in records]] - mean) / std
        return records, Encoded(records, tokenizer, label_map, 128, 0), torch.tensor(aux, dtype=torch.float32)

    train_records, train_set, train_aux = split_data("train")
    dev_records, dev_set, dev_aux = split_data("dev")

    model = UtteranceClassifier(MODEL, len(names), aux_dim=FEATURE_DIM).to(device)
    hidden = model.encoder.config.hidden_size
    assert model.classifier.weight.shape == (len(names), hidden + FEATURE_DIM)
    print(f"classifier weight {tuple(model.classifier.weight.shape)} = 7 x ({hidden} <s> + {FEATURE_DIM} DeepMoji)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for batch in DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(SEED)):
        aux = train_aux[batch["index"]].to(device)
        loss = loss_fn(model(batch["input_ids"].to(device), batch["attention_mask"].to(device), aux),
                       batch["labels"].to(device))
        loss.backward()
        grad = model.classifier.weight.grad
        encoder_grad = sum(p.grad.norm() ** 2 for p in model.encoder.parameters() if p.grad is not None) ** 0.5
        print(f"  loss {loss.item():.3f}  classifier grad norm: <s> part {grad[:, :hidden].norm():.2f}, "
              f"DeepMoji part {grad[:, hidden:].norm():.2f}; encoder grad norm {encoder_grad:.2f}")
        assert torch.isfinite(loss) and grad[:, hidden:].abs().sum() > 0 and encoder_grad > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.eval()
    predicted = [None] * len(dev_set)
    with torch.no_grad():
        for batch in DataLoader(dev_set, batch_size=32, collate_fn=collate):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                           dev_aux[batch["index"]].to(device))
            for i, k in zip(batch["index"].tolist(), logits.argmax(-1).tolist()):
                predicted[i] = names[k]
    metrics = evaluate(dev_records, predicted, names)
    print(f"dev[:{LIMIT}] via score.py: weighted F1 {100 * metrics['weighted_f1']:.2f}, "
          f"macro F1 {100 * metrics['macro_f1']:.2f} (64 utterances after 4 steps: not meaningful)")

    try:
        model(batch["input_ids"].to(device), batch["attention_mask"].to(device), None)
        raise SystemExit("shape guard did not fire for missing aux_features")
    except ValueError as error:
        print(f"shape guard ok: {error}")
    print("fusion smoke test passed")


if __name__ == "__main__":
    main()
