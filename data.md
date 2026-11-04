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
    train.jsonl             # Official train membership, sorted by dialogue/utterance
    train_clean.jsonl       # Recommended training targets AND context
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
| Official train (`train.jsonl`) | 9,989 | 1,038 | Original benchmark copy |
| Clean train (`train_clean.jsonl`) | 9,988 | 1,038 | Training and training-context construction |
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

Neutral constitutes 47.15% of clean training records; fear is 2.68% and disgust is 2.71%. Planned evaluation uses weighted F1 and macro F1, supported by per-class scores and a confusion matrix. No trained-model performance is claimed in this milestone.

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

The clean training file removes training records matching a dev/test source-time key. The rule uses source metadata and text, not emotion labels; it excludes exactly `train:12:7` in this snapshot. Original raw CSVs and `train.jsonl` remain intact for traceability and explicitly labeled official-protocol comparisons.

**Use `train_clean.jsonl` for both training examples and the history from which training contexts are built.** Excluding the record only as a target while retaining it as context would still expose its text. Use the unchanged dev and test files. The clean training source-time keys have zero overlap with held-out keys under this audit. This check does not establish absence of paraphrases, scene-level dependencies, or every possible form of contamination.

## Planned experimental use

1. Build inputs only from a record's text, speaker information, and preceding retained turns in the same `(split, dialogue_id)` group. History windows count observed turns, not missing numeric IDs.
2. Keep gold emotion and sentiment annotations out of model inputs and context strings.
3. Determine the majority class from clean training records. In every neural condition, fine-tune the same RoBERTa-base checkpoint end to end, with the same shared training settings.
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
