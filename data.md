# MELD Text Data for CIS 5300 Milestone 1

## Source and collection

- Dataset: MELD, the original multiparty version; not MELD-Dyadic or EmoryNLP.
- Dataset paper: Poria et al. (2019), [MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/).
- Repository: [declare-lab/MELD](https://github.com/declare-lab/MELD).
- Pinned revision: [`e8cedf27b5d2877e198332c957127e16eb214afe`](https://github.com/declare-lab/MELD/tree/e8cedf27b5d2877e198332c957127e16eb214afe).
- Collection date: November 3, 2026, America/New_York.
- Downloaded files: `data/MELD/train_sent_emo.csv`, `dev_sent_emo.csv`, and `test_sent_emo.csv` at the pinned revision. Their full source URLs and SHA-256 checksums appear in `data/manifest.json`.
- Scope: English utterance text and existing emotion annotations. No new annotation, synthetic examples, audio/video downloads, or Twitter corpus collection was performed.

The upstream CSVs and repository license are retained verbatim. Processed files preserve the original text, speakers, labels, and dialogue/utterance IDs; no lowercasing, punctuation normalization, or tokenization is applied. Consequently, any upstream text-encoding artifacts are also retained. The seven-way emotion task is separate from MELD's three-way sentiment task; sentiment annotations are not features or targets here.

## Files

```text
data/
  raw/
    train_sent_emo.csv       # Original bytes from the pinned repository
    dev_sent_emo.csv
    test_sent_emo.csv
  processed/
    train.jsonl             # Official train membership, sorted by dialogue/utterance (main protocol)
    train_clean.jsonl       # Optional copy without one exact-key match (supplementary)
    dev.jsonl               # Official dev membership, unchanged
    test.jsonl              # Official test membership, unchanged
  label_map.json
  label_distribution.csv
  statistics.json
  excluded_training_records.json
  manifest.json
  SOURCE_LICENSE            # Verbatim upstream repository license
```

The supplied `meld_text_data.tar.gz` contains the complete `data/` directory and this `data.md`, rather than a sample.

## Sizes and label distributions

These counts are measured from the downloaded CSVs and generated JSONL files.

| Split | Utterances | Distinct dialogue IDs | Role |
|---|---:|---:|---|
| Official train (`train.jsonl`) | 9,989 | 1,038 | Main protocol: training and training-context construction |
| Exact-key-filtered train (`train_clean.jsonl`) | 9,988 | 1,038 | Supplementary sensitivity check |
| Dev (`dev.jsonl`) | 1,109 | 114 | Model/configuration selection |
| Test (`test.jsonl`) | 2,610 | 280 | Final evaluation |

The repository description reports 1,039 training dialogues. In this pinned CSV, dialogue ID 60 is absent, and only 1,038 distinct training IDs occur. We report the observed count and do not invent an empty dialogue. Dialogue IDs are split-local: the same integer may appear in different splits without referring to the same dialogue.

| Emotion | Label ID | Official train | Clean train | Dev | Test |
|---|---:|---:|---:|---:|---:|
| neutral | 0 | 4,710 | 4,709 | 470 | 1,256 |
| surprise | 1 | 1,205 | 1,205 | 150 | 281 |
| fear | 2 | 268 | 268 | 40 | 50 |
| sadness | 3 | 683 | 683 | 111 | 208 |
| joy | 4 | 1,743 | 1,743 | 163 | 402 |
| disgust | 5 | 271 | 271 | 22 | 68 |
| anger | 6 | 1,109 | 1,109 | 153 | 345 |
| **Total** | | **9,989** | **9,988** | **1,109** | **2,610** |

Neutral constitutes 47.15% of official training records (the same to two decimals in the filtered copy); fear is 2.68% and disgust is 2.71%. Planned evaluation uses weighted F1 and macro F1, supported by per-class scores and a confusion matrix. No trained-model performance is claimed in this milestone.

## Format and real example

Raw files are CSVs with a header. They also contain source season/episode, timestamps, and sentiment metadata. Processed files are UTF-8 JSON Lines: one JSON object per utterance, with no header. All records follow this schema:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Globally unique `split:dialogue_id:utterance_id` |
| `split` | string | Original split: `train`, `dev`, or `test` |
| `dialogue_id` | integer | Original split-local dialogue ID |
| `utterance_id` | integer | Original turn ID within the dialogue |
| `speaker` | string | Speaker identity from the CSV |
| `text` | string | Unmodified utterance text |
| `label` | string | Gold emotion label; never an input feature |
| `label_id` | integer | Seven-way target according to `label_map.json` |

This is an actual retained training record, not a constructed illustration:

```json
{"id":"train:0:2","split":"train","dialogue_id":0,"utterance_id":2,"speaker":"Chandler","text":"That I did. That I did.","label":"neutral","label_id":0}
```

Its model input is the text and speaker, optionally with preceding retained turns of the same dialogue; its target is `neutral` (ID 0).

## Integrity checks and overlap handling

Preparation validates the required columns, known labels, nonempty text/speaker fields, unique `(Dialogue_ID, Utterance_ID)` pairs within each split, and exact equality of processed text/speaker/labels with their source rows. SHA-256 checksums cover raw and processed artifacts. No rows fail these basic checks.

Some original turn IDs have gaps: 72 training dialogues, 2 dev dialogues, and 2 test dialogues have IDs that do not start at zero or are not contiguous. Preserve those IDs and iterate over the observed sorted turns. Do not assume `utterance_id - 1` identifies an existing previous turn, and do not fill missing turns with invented text.

Exact text repetition alone is not treated as a duplicate example: short responses can legitimately recur in different scenes or contexts. We additionally compare the tuple `(Season, Episode, StartTime, EndTime, Speaker, Utterance)`, which identifies source-time occurrences more precisely.

| Official split pair | Shared exact text strings | Shared source-time/speaker/text keys |
|---|---:|---:|
| Train / dev | 73 | 0 |
| Train / test | 136 | 1 |
| Dev / test | 43 | 0 |

The single matching source-time item is `train:12:7` / `test:85:0`: Joey's utterance in season 6, episode 4, from `00:10:18,743` to `00:10:21,536`. The source records also share their neutral annotation. Full details are recorded in `excluded_training_records.json`.

### Suspected scene-level overlap (heuristic audit)

The exact key above requires identical season, episode, timestamps, speaker, and text, so it misses the same scene when MELD segments it differently or records inconsistent metadata. A second, heuristic audit flagged held-out utterances whose normalized text (lowercase alphanumerics, at least 40 characters) is contained in a training utterance or vice versa, or whose speaker, end timestamp, and text prefix match a training utterance. These rules flag individual utterances; we then manually reviewed each flagged scene side by side with its training counterpart and added neighboring turns from the same source interval that the rules missed (for example, short lines or re-segmented turns). The rules found no dev matches; the reviewed test overlaps are:

| Test items | Training items | Observation |
|---|---|---|
| `test:84:0–2` (whole dialogue) | `train:12:4–6` | Same speakers and text, same or near-identical timestamps; test metadata says season 1, episode 5 instead of 6/4, and all three emotion labels differ |
| `test:85:0–5` (whole dialogue) | `train:12:7–11` | The same scene segmented differently; only `85:0` / `train:12:7` matches the exact key |
| `test:155:0–1` | `train:51:8`, `train:51:10` | Season 3, episode 15; near-identical text and overlapping source intervals. The first pair shares an end time and was flagged by the rules; the second shares a start time and was added on manual review |

This is roughly 11 test utterances in 3 test dialogues (about 0.4% of test). It is a list of suspected same-source overlap found by heuristic rules plus manual review of the flagged scenes, not a complete inventory: paraphrased or more heavily re-segmented overlap may remain. A recurring line such as `train:776:4` / `test:225:0` from a different time in the same episode was judged a legitimate repetition.

### Training protocol

**Main protocol: use the official splits** (`train.jsonl`, `dev.jsonl`, `test.jsonl`) so results are comparable with published MELD numbers, and report the suspected overlap above as a known limitation.

`train_clean.jsonl` is an optional sensitivity check. It removes training records matching a dev/test exact source-time key; the rule uses source metadata and text, not emotion labels, and excludes exactly `train:12:7` in this snapshot. When it is used, it must supply both the training targets and the history from which training contexts are built, because excluding a record only as a target while retaining it as context would still expose its text. Its exact source-time keys have zero overlap with held-out keys, but this statement is limited to that strict key: the scene-level overlaps listed above remain in it (for example `train:12:4–6` and `train:12:8–11`), so it must not be described as fully deduplicated. Dev and test files are never modified.

## Planned experimental use

1. Build inputs only from a record's text, speaker information, and preceding retained turns in the same `(split, dialogue_id)` group. History windows count observed turns, not missing numeric IDs.
2. Keep gold emotion and sentiment annotations out of model inputs and context strings.
3. Determine the majority class from the training file of the protocol in use (official `train.jsonl` by default). In every neural condition, fine-tune the same RoBERTa-base checkpoint end to end, with the same shared training settings.
4. Cross history off/on with three auxiliary-feature choices: none, a frozen randomly initialized DeepMoji architecture, and frozen pretrained DeepMoji. This gives six neural conditions. All history-enabled conditions use the same concatenation architecture; both auxiliary branches use only the current utterance and share tokenizer, feature dimension, and preprocessing. Neither the random encoder nor the pretrained DeepMoji encoder is trained on MELD. The original emoji-pretraining corpus is not required or supplied.
5. Run at least three matched random seeds; report weighted and macro F1 as mean +/- sample standard deviation. Use dev for configuration/early-stopping decisions and reserve test for final evaluation. If paired bootstrap is used, sample whole dialogues and pair predictions by stable sample IDs.
6. Report which training protocol was used, retain negative results, and interpret pretrained-versus-random differences without assuming they isolate emoji supervision from Twitter-domain exposure. Full specifications and implementation priorities appear in `experiment_plan.md`; published scores are not our own reproduced results.

## Reproduce and verify

From this milestone directory, using Python 3 and only its standard library:

```bash
python3 prepare_data.py --download --archive
python3 prepare_data.py --verify-only
```

The first command retrieves the pinned files, regenerates all processed data and audit metadata, checks consistency, and packages the archive. The second performs offline checksum and source-to-record validation without rewriting files. Extract the supplied data archive with:

```bash
tar -xzf meld_text_data.tar.gz
```

The upstream repository license is included as `data/SOURCE_LICENSE`; the dataset paper and source repository should be credited in the project report.
