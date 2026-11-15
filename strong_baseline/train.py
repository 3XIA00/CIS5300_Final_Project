"""Fine-tune RoBERTa-base on the current utterance only (Milestone 2 strong baseline, condition B).

Run from the repository root:

    python -m strong_baseline.train --seed 13
    python -m strong_baseline.train --seed 13 --limit 64 --epochs 1   # quick smoke test

Training follows experiment_plan.md: AdamW, lr 2e-5, batch 16, weight decay 0.01,
at most 5 epochs, linear decay after 10% warmup, unweighted cross entropy. After
every epoch the model is scored on dev; the checkpoint with the best dev weighted
F1 is kept (ties keep the earlier epoch) and training stops after 2 epochs
without improvement. The best checkpoint then predicts dev and test, and the
predictions are scored with score.py on the full test set and with the
suspected-overlap ids excluded.

Writes (paths per interface.md):
    output/roberta-base/seed<N>/{dev,test}.jsonl   predictions
    output/roberta-base/seed<N>/run.json           config, per-epoch dev scores, test scores,
                                                   wall-clock time, peak GPU memory, versions
    checkpoints/roberta-base/seed<N>/              best weights (git-ignored)
"""

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from strong_baseline.data import ROOT, build_input, label_names, load_label_map, load_split, previous_turns
from strong_baseline.model import UtteranceClassifier

sys.path.insert(0, str(ROOT))
import score  # noqa: E402  (the project's only scorer; see interface.md)

OVERLAP_IDS = ROOT / "eval" / "suspected_overlap_test_ids.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="roberta-base", help="Hugging Face model name or local path")
    p.add_argument("--system", default="roberta-base", help="name used under output/ and checkpoints/")
    p.add_argument("--seed", type=int, default=13, help="plan seeds: 13, 42, 2026 (then 31415, 27182)")
    p.add_argument("--train-split", default="train", choices=["train", "train_clean"],
                   help="train = official split (main protocol); train_clean = sensitivity check")
    p.add_argument("--history", type=int, default=0, help="previous turns to include (0 for condition B)")
    p.add_argument("--max-length", type=int, default=128, help="max RoBERTa tokens per input")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2, help="stop after this many epochs without dev gain")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true", help="mixed precision on CUDA (faster, less memory)")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--limit", type=int, help="use only the first N records of each split (smoke tests)")
    p.add_argument("--output-dir", help="default: output/<system>/seed<N>")
    p.add_argument("--checkpoint-dir", help="default: checkpoints/<system>/seed<N>")
    p.add_argument("--no-checkpoint", action="store_true", help="do not save weights to disk")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Encoded(torch.utils.data.Dataset):
    """Tokenized inputs for one split; tokenization happens once, up front."""

    def __init__(self, records, tokenizer, label_map, max_length, n_history):
        history = previous_turns(records, n_history)
        pairs = [build_input(r, history[r["id"]]) for r in records]
        targets = [t for t, _ in pairs]
        contexts = [c for _, c in pairs]
        kwargs = dict(truncation="only_second" if n_history else True, max_length=max_length)
        if n_history:
            self.enc = tokenizer(targets, contexts, **kwargs)
        else:
            self.enc = tokenizer(targets, **kwargs)
        self.ids = [r["id"] for r in records]
        self.labels = [label_map[r["label"]] for r in records]
        full = tokenizer(targets) if not n_history else tokenizer(targets, contexts)
        lengths = sorted(len(x) for x in full["input_ids"])
        self.length_stats = {
            "max": lengths[-1],
            "p95": lengths[int(0.95 * (len(lengths) - 1))],
            "truncated": sum(n > max_length for n in lengths),
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i], "index": i}


def make_collate(pad_id):
    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[row, :n] = torch.tensor(item["input_ids"])
            mask[row, :n] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": torch.tensor([item["labels"] for item in batch]),
                "index": torch.tensor([item["index"] for item in batch])}
    return collate


@torch.no_grad()
def predict(model, loader, device, names, fp16):
    """Return predicted label names in dataset order."""
    model.eval()
    out = [None] * len(loader.dataset)
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for i, k in zip(batch["index"].tolist(), logits.argmax(-1).tolist()):
            out[i] = names[k]
    return out


def evaluate(records, predicted, names, exclude=frozenset()):
    gold = {r["id"]: r["label"] for r in records}
    pred = {r["id"]: label for r, label in zip(records, predicted)}
    g, p = score.align(gold, pred, exclude)
    return score.compute_metrics(g, p, names)


def model_revision(model_name):
    """Commit hash of the downloaded Hugging Face checkpoint (None for a local folder)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_name, "config.json")
    except Exception:
        return None
    if isinstance(path, str) and "/snapshots/" in path:
        return path.split("/snapshots/")[1].split("/")[0]
    return None


def code_version():
    """Git commit of this code and whether it had uncommitted changes (None outside a git checkout)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "strong_baseline", "score.py"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "uncommitted_changes": dirty}


def summary(metrics):
    return {k: round(100 * metrics[k], 2) for k in ("weighted_f1", "macro_f1", "accuracy")} | {"n": metrics["n"]}


def write_predictions(path, records, predicted):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, label in zip(records, predicted):
            handle.write(json.dumps({"id": record["id"], "label": label}) + "\n")


def main(argv=None):
    args = parse_args(argv)
    run_name = f"seed{args.seed}"
    output_dir = Path(args.output_dir or ROOT / "output" / args.system / run_name)
    checkpoint_dir = Path(args.checkpoint_dir or ROOT / "checkpoints" / args.system / run_name)
    device = pick_device(args.device)
    fp16 = args.fp16 and device.type == "cuda"
    set_seed(args.seed)

    label_map = load_label_map()
    names = label_names(label_map)
    splits = {name: load_split(name) for name in (args.train_split, "dev", "test")}
    if args.limit:
        splits = {name: records[:args.limit] for name, records in splits.items()}
    train_records, dev_records, test_records = splits[args.train_split], splits["dev"], splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {name: Encoded(records, tokenizer, label_map, args.max_length, args.history)
            for name, records in splits.items()}
    for name, dataset in data.items():
        print(f"{name}: {len(dataset)} utterances, tokens {dataset.length_stats}", flush=True)

    collate = make_collate(tokenizer.pad_token_id)
    shuffle_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(data[args.train_split], batch_size=args.batch_size, shuffle=True,
                              generator=shuffle_gen, collate_fn=collate)
    dev_loader = DataLoader(data["dev"], batch_size=args.eval_batch_size, collate_fn=collate)
    test_loader = DataLoader(data["test"], batch_size=args.eval_batch_size, collate_fn=collate)

    model = UtteranceClassifier(args.model, len(names), dropout=args.dropout).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    loss_fn = torch.nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, best_f1, best_epoch, best_state, bad_epochs = [], -1.0, None, None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start, total_loss = time.time(), 0.0
        for batch in train_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = loss_fn(logits.float(), batch["labels"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * len(batch["labels"])

        dev = summary(evaluate(dev_records, predict(model, dev_loader, device, names, fp16), names))
        improved = dev["weighted_f1"] > best_f1  # strict: ties keep the earlier checkpoint
        history.append({"epoch": epoch, "train_loss": round(total_loss / len(data[args.train_split]), 4),
                        "dev": dev, "seconds": round(time.time() - epoch_start, 1), "best": improved})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']}  dev weighted F1 {dev['weighted_f1']}  "
              f"macro F1 {dev['macro_f1']}{'  (best)' if improved else ''}", flush=True)
        if improved:
            best_f1, best_epoch, bad_epochs = dev["weighted_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"stopping: no dev improvement for {bad_epochs} epochs", flush=True)
                break
    train_seconds = time.time() - started

    model.load_state_dict(best_state)
    if not args.no_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / "model.pt")
        (checkpoint_dir / "config.json").write_text(json.dumps(
            {"model": args.model, "num_labels": len(names), "labels": names, "dropout": args.dropout,
             "max_length": args.max_length, "history": args.history}, indent=2) + "\n", encoding="utf-8")

    dev_pred = predict(model, dev_loader, device, names, fp16)
    test_pred = predict(model, test_loader, device, names, fp16)
    write_predictions(output_dir / "dev.jsonl", dev_records, dev_pred)
    write_predictions(output_dir / "test.jsonl", test_records, test_pred)

    exclude = score.load_exclude_ids(OVERLAP_IDS) & {r["id"] for r in test_records}
    results = {
        "dev": summary(evaluate(dev_records, dev_pred, names)),
        "test": summary(evaluate(test_records, test_pred, names)),
        "test_excluding_suspected_overlap": summary(evaluate(test_records, test_pred, names, exclude)),
    }
    run = {
        "system": args.system,
        "config": vars(args),
        "best_epoch": best_epoch,
        "epochs": history,
        "results": results,
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "token_lengths": {name: d.length_stats for name, d in data.items()},
        "model_revision": model_revision(args.model),
        "code": code_version(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "limit": args.limit,
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    print(f"\nbest epoch {best_epoch}; trained {train_seconds / 60:.1f} min")
    for name, r in results.items():
        print(f"{name:<34} weighted F1 {r['weighted_f1']:6.2f}  macro F1 {r['macro_f1']:6.2f}  n={r['n']}")
    print(f"predictions and run.json in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
