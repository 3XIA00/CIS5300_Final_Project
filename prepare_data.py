#!/usr/bin/env python3
"""Collect the pinned official MELD text splits and prepare the M1 data archive.

Uses only Python's standard library. Raw CSV files are retained byte for byte;
processed JSONL records preserve the source text, speakers, labels, and splits.
"""

import argparse
import csv
import hashlib
import io
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMMIT = "e8cedf27b5d2877e198332c957127e16eb214afe"
BASE_URL = f"https://raw.githubusercontent.com/declare-lab/MELD/{COMMIT}"
LABELS = {
    "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
    "joy": 4, "disgust": 5, "anger": 6,
}
SPLITS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_key(row):
    return tuple(row[field] for field in
                 ("Season", "Episode", "StartTime", "EndTime", "Speaker", "Utterance"))


def source_keys(rows):
    return {source_key(row) for row in rows}


def download(relative_url, target, force):
    if force or not target.exists():
        request = urllib.request.Request(
            f"{BASE_URL}/{relative_url}", headers={"User-Agent": "CIS5300-M1-data-collection"}
        )
        with urllib.request.urlopen(request, timeout=40) as response:
            payload = response.read()
        target.write_bytes(payload)


def prepare(force=False):
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    download("LICENSE", DATA / "SOURCE_LICENSE", force)
    statistics, rows_by_split, records_by_split = {}, {}, {}
    provenance = []
    for split, filename in SPLITS.items():
        raw = DATA / "raw" / filename
        download(f"data/MELD/{filename}", raw, force)
        rows = list(csv.DictReader(io.StringIO(raw.read_text(encoding="utf-8-sig"))))
        required = {"Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Unexpected schema or empty split: {raw}")
        rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))
        records, seen, dialogues = [], set(), defaultdict(list)
        for row in rows:
            did, uid = int(row["Dialogue_ID"]), int(row["Utterance_ID"])
            key = (did, uid)
            if key in seen:
                raise ValueError(f"Duplicate utterance identifier in {split}: {key}")
            seen.add(key)
            if not row["Utterance"].strip() or not row["Speaker"].strip():
                raise ValueError(f"Empty text or speaker in {split}: {key}")
            label = row["Emotion"]
            if label not in LABELS:
                raise ValueError(f"Unknown label: {label}")
            record = {
                "id": f"{split}:{did}:{uid}", "split": split,
                "dialogue_id": did, "utterance_id": uid,
                "speaker": row["Speaker"], "text": row["Utterance"],
                "label": label, "label_id": LABELS[label],
            }
            records.append(record)
            dialogues[did].append(uid)
        gaps = {str(did): ids for did, ids in dialogues.items() if ids != list(range(len(ids)))}
        output = DATA / "processed" / f"{split}.jsonl"
        output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        counts = Counter(r["label"] for r in records)
        statistics[split] = {
            "utterances": len(records), "dialogues": len(dialogues),
            "label_counts": {label: counts[label] for label in LABELS},
            "noncontiguous_dialogues": gaps,
            "duplicate_dialogue_utterance_ids": 0,
            "empty_text_or_speaker": 0,
            "unrecognized_labels": 0,
            "unique_exact_texts": len({r["text"] for r in records}),
        }
        provenance.append({"split": split, "url": f"{BASE_URL}/data/MELD/{filename}",
                           "path": str(raw.relative_to(ROOT)), "sha256": sha256(raw)})
        rows_by_split[split], records_by_split[split] = rows, records

    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_text = {r["text"] for r in records_by_split[left]}
        right_text = {r["text"] for r in records_by_split[right]}
        shared_text = left_text & right_text
        # Source-time identities are stronger than string equality: short replies
        # such as "Yeah." can occur in multiple, otherwise distinct conversations.
        overlap[f"{left}_{right}"] = {
            "shared_exact_text_strings": len(shared_text),
            "shared_source_time_speaker_text_keys": len(source_keys(rows_by_split[left]) & source_keys(rows_by_split[right])),
        }

    # Keep the official benchmark files intact, but provide a clean training
    # protocol that never learns from the same source-time item as dev or test.
    heldout_keys = source_keys(rows_by_split["dev"]) | source_keys(rows_by_split["test"])
    excluded = []
    for row in rows_by_split["train"]:
        if source_key(row) in heldout_keys:
            excluded.append({
                "id": f"train:{row['Dialogue_ID']}:{row['Utterance_ID']}",
                "reason": "same season, episode, timestamps, speaker and text as a held-out utterance",
                "source_row": row,
            })
    excluded_ids = {row["id"] for row in excluded}
    clean = [r for r in records_by_split["train"] if r["id"] not in excluded_ids]
    (DATA / "processed" / "train_clean.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean), encoding="utf-8"
    )
    write_json(DATA / "excluded_training_records.json", excluded)
    clean_counts = Counter(r["label"] for r in clean)
    clean_protocol = {
        "train_file": "processed/train_clean.jsonl",
        "train_utterances": len(clean),
        "train_dialogues": len({r["dialogue_id"] for r in clean}),
        "train_label_counts": {label: clean_counts[label] for label in LABELS},
        "excluded_train_ids": sorted(excluded_ids),
        "dev_file": "processed/dev.jsonl", "test_file": "processed/test.jsonl",
        "rule_uses_emotion_labels": False,
        "removed_items_must_not_be_used_as_training_context": True,
    }

    write_json(DATA / "label_map.json", LABELS)
    write_json(DATA / "statistics.json", {"splits": statistics, "cross_split_overlap": overlap, "clean_protocol": clean_protocol})
    with (DATA / "label_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "label_id", "train_official", "train_clean", "dev", "test"])
        for label, lid in LABELS.items():
            writer.writerow([label, lid, statistics["train"]["label_counts"][label], clean_counts[label]]
                            + [statistics[s]["label_counts"][label] for s in ("dev", "test")])
    files = {str(p.relative_to(DATA)): sha256(p) for p in sorted(DATA.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(DATA / "manifest.json", {
        "dataset": "MELD, official multiparty text splits", "collection_date": "2026-11-03",
        "repository": "https://github.com/declare-lab/MELD", "commit": COMMIT,
        "sources": provenance, "files_sha256": files,
    })
    summary = {
        split: {"utterances": st["utterances"], "dialogues": st["dialogues"],
                "dialogues_with_noncontiguous_ids": len(st["noncontiguous_dialogues"])}
        for split, st in statistics.items()
    }
    print(json.dumps({"splits": summary, "cross_split_overlap": overlap, "clean_protocol": clean_protocol}, indent=2))


def verify():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files_sha256"].items():
        if sha256(DATA / relative) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    for split in SPLITS:
        records = [json.loads(line) for line in (DATA / "processed" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = list(csv.DictReader(io.StringIO((DATA / "raw" / SPLITS[split]).read_text(encoding="utf-8-sig"))))
        originals = {(int(r["Dialogue_ID"]), int(r["Utterance_ID"])): r for r in rows}
        if len(records) != len(rows):
            raise ValueError(f"Row count mismatch: {split}")
        for record in records:
            row = originals[(record["dialogue_id"], record["utterance_id"])]
            if (record["text"], record["speaker"], record["label"]) != (row["Utterance"], row["Speaker"], row["Emotion"]):
                raise ValueError(f"Source content changed: {record['id']}")
            if record["label_id"] != LABELS[record["label"]]:
                raise ValueError(f"Wrong label mapping: {record['id']}")
    excluded = json.loads((DATA / "excluded_training_records.json").read_text(encoding="utf-8"))
    excluded_ids = {r["id"] for r in excluded}
    official = [json.loads(line) for line in (DATA / "processed" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    clean = [json.loads(line) for line in (DATA / "processed" / "train_clean.jsonl").read_text(encoding="utf-8").splitlines()]
    if clean != [r for r in official if r["id"] not in excluded_ids]:
        raise ValueError("Clean training file does not match its documented exclusions")
    train_rows = list(csv.DictReader(io.StringIO((DATA / "raw" / SPLITS["train"]).read_text(encoding="utf-8-sig"))))
    clean_rows = [r for r in train_rows if f"train:{r['Dialogue_ID']}:{r['Utterance_ID']}" not in excluded_ids]
    heldout_rows = []
    for split in ("dev", "test"):
        heldout_rows.extend(csv.DictReader(io.StringIO((DATA / "raw" / SPLITS[split]).read_text(encoding="utf-8-sig"))))
    if source_keys(clean_rows) & source_keys(heldout_rows):
        raise ValueError("Source-time overlap remains between clean training and held-out data")
    print(f"Verified {len(manifest['files_sha256'])} checksums and all source-to-JSONL records.")


def archive():
    target = ROOT / "meld_text_data.tar.gz"
    with tarfile.open(target, "w:gz") as handle:
        handle.add(DATA, arcname="data")
        if (ROOT / "data.md").exists():
            handle.add(ROOT / "data.md", arcname="data.md")
    with tarfile.open(target, "r:gz") as handle:
        print(f"Archive: {target.name}; {len(handle.getnames())} entries; {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Redownload the pinned original files.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing data without writing files or accessing the network.")
    parser.add_argument("--archive", action="store_true", help="Also package the data and data.md as a gzipped tar archive.")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        prepare(args.download)
        verify()
    if args.archive:
        archive()
