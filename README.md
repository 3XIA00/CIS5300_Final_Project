# CIS 5300 Milestone 1 Deliverables

## What to submit

- `writeup.pdf`: the Milestone 1 report (two pages). `writeup.docx` and `writeup.md` are its editable sources and carry the same text. It reviews DeepMoji, EmoBERTa, and CoMPM and describes the collected MELD data.
- `data.md`: data source, schema, a real example, measured split/class counts, quality checks, suspected cross-split overlap, and the training protocol.
- `meld_text_data.tar.gz`: the complete raw and processed text data, provenance, label map, statistics, source license, and `data.md` (identical to the copy in this folder).

## Other files

- `experiment_plan.md`: supplementary six-condition design incorporating the literature review and Ethan's meeting feedback: shared context architecture, fine-tuned RoBERTa, frozen random/pretrained controls, seed matching, uncertainty reporting, implementation priorities, and interpretation of negative results.
- `prepare_data.py`: standard-library-only collection, preparation, archive, and offline verification commands.
- `data/`: unpacked data corresponding to the archive.
- `archive/`: earlier drafts kept for reference only; not part of the submission.

The writeup distinguishes published results from future project experiments. It does not claim that a baseline, extension, or TorchMoji compatibility test has been completed.

## Use the data

**Main protocol: the official splits** `data/processed/train.jsonl` (9,989 utterances), `dev.jsonl` (1,109), and `test.jsonl` (2,610), so results stay comparable with published MELD numbers.

`data/processed/train_clean.jsonl` (9,988) is an optional sensitivity check that removes one training record whose exact season/episode/timestamps/speaker/text key also appears in test. It is not fully deduplicated: `data.md` lists further suspected scene-level overlap between train and test that this strict key does not catch.

```bash
python prepare_data.py --verify-only
```

To redownload the pinned originals and rebuild the package:

```bash
python prepare_data.py --download --archive
```

The data files must stay byte-identical to the checksums in `data/manifest.json`. `.gitattributes` turns off Git's line-ending conversion for `data/` so verification also passes on Windows.

## Course submission

The course page requests a report of no more than two pages, `data.md`, and a gzipped data archive. `writeup.pdf` was exported from `writeup.docx` and checked at two pages; re-export after any edit and re-check the page count. Keep `data.md` separate from the report.

The files incorporate the supplied meeting summary from Ethan; no meeting date, future appointment, or team-member names have been invented. Follow the course requirement to schedule the next mentor meeting separately.
