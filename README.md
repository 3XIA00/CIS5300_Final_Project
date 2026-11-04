# CIS 5300 Milestone 1 Deliverables

- `writeup.md`: Revised English M1 writeup with two paragraphs each on DeepMoji, DialogueRNN, and EmoBERTa, the collected-data description, and a concise mentor-informed plan. MELD is additionally cited as the dataset source.
- `experiment_plan.md`: Supplementary six-condition design incorporating the paper recommendations and Ethan's meeting feedback: shared context architecture, fine-tuned RoBERTa, frozen random/pretrained controls, seed matching, uncertainty reporting, implementation priorities, and interpretation of negative results.
- `data.md`: Data source, schema, actual examples, measured split/class counts, quality checks, and the documented clean-training protocol.
- `meld_text_data.tar.gz`: Complete raw and processed text data, provenance, label map, statistics, source license, and `data.md`.
- `prepare_data.py`: Standard-library-only collection, preparation, archive, and offline verification commands.
- `data/`: Unpacked data corresponding to the archive.

The writeup distinguishes published results from future project experiments. It does not claim that a baseline, extension, or TorchMoji compatibility test has been completed.

The updated core reading selection follows `../milestone1_papers.md`: EmoBERTa replaces DialogueGCN as a full review because it directly supports the initial concatenation implementation. Supplementary references, checked distinctions, and limits of the random-encoder control appear in `experiment_plan.md`.

## Use the data

Use `data/processed/train_clean.jsonl`, `dev.jsonl`, and `test.jsonl`. The clean training file has 9,988 utterances because one training source-time occurrence is also present in test. The original 9,989-record training split is retained separately and is not silently overwritten.

```bash
python3 prepare_data.py --verify-only
```

To redownload the pinned originals and rebuild the package:

```bash
python3 prepare_data.py --download --archive
```

## Course submission

The full course page requests a report of no more than two pages, `data.md`, and a gzipped data archive. The report is delivered as Markdown as requested. A temporary reference layout using US Letter, 11-point Times body text, and 0.7-inch left/right margins was rendered and visually checked at two pages; other export settings can change pagination. Export `writeup.md` to the required submission format and keep `data.md` separate from that report. Retain the data archive for the data-submission component.

The files incorporate the supplied meeting summary from Ethan; no meeting date, future appointment, or team-member names have been invented. Follow the course requirement to schedule the next mentor meeting separately.
