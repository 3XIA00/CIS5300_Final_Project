# Strong Baseline: RoBERTa-base on the Current Utterance

This is condition **B** of `experiment_plan.md`: RoBERTa-base, fine-tuned end to end, classifying each utterance from its own text and speaker name, without dialogue history or DeepMoji features. It is the starting point for Milestone 3, which adds history (C) and DeepMoji features (D, R, CD, CR) to the same code.

## Model

Each utterance $i$ is written as `Speaker: text` (the speaker-aware input of EmoBERTa; Kim and Vossen, 2021) and tokenized with the RoBERTa tokenizer (at most 128 tokens; the longest training utterance has 70 words, so nothing is truncated in practice, and `run.json` records the exact count). Let $h_i \in \mathbb{R}^{768}$ be the final-layer hidden state of the `<s>` token. A linear layer gives the class distribution over the $K = 7$ emotions:

$$
p(y \mid x_i) = \mathrm{softmax}(W h_i + b), \qquad W \in \mathbb{R}^{7 \times 768}
$$

All RoBERTa weights and the classifier are trained with unweighted cross entropy over the training set $\mathcal{D}$:

$$
\mathcal{L} = -\frac{1}{|\mathcal{D}|} \sum_{i \in \mathcal{D}} \log p(y_i \mid x_i)
$$

The prediction is $\hat{y}_i = \arg\max_y p(y \mid x_i)$.

## Training

Settings follow `experiment_plan.md`:

| Setting | Value |
|---|---|
| Checkpoint | `roberta-base` (Liu et al., 2019); exact revision recorded in `run.json` |
| Training data | `data/processed/train.jsonl`, official split, 9,989 utterances |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 (not on biases and LayerNorm) |
| Batch size | 16 |
| Schedule | linear warmup over the first 10% of steps, then linear decay to 0 |
| Epochs | at most 5; stop after 2 epochs without dev improvement |
| Model selection | best dev weighted F1 (ties keep the earlier epoch) |
| Other | dropout 0.1 before the classifier, gradient clipping at norm 1.0 |
| Seeds | 13, 42, 2026, 31415, 27182 (all five run; see Results); the seed fixes initialization of the classifier, dropout and data order |

Test labels are never used for training or model selection.

## Code

```text
strong_baseline/
  data.py      loading records, grouping dialogues, build_input() (no torch needed)
  model.py     RoBERTa + linear classifier on <s>; aux_dim hook for DeepMoji features in M3
  train.py     training loop, dev selection, predictions, scores, run.json
  predict.py   regenerate predictions from a saved checkpoint
  colab.ipynb  run everything on a Colab GPU and save the results
  requirements.txt
```

Inputs are built only in `data.build_input`. History (`--history N`) is already supported for Milestone 3; it takes previous turns in observed order within the dialogue, never by arithmetic on utterance ids, and drops the oldest turns first when the input is too long. The Milestone 2 baseline uses `--history 0` (the default).

## How to run

Training needs a GPU (Colab, or a school machine). From the repository root:

```bash
pip install -r strong_baseline/requirements.txt
python -m strong_baseline.train --seed 13 --limit 64 --epochs 1    # smoke test, about a minute
python -m strong_baseline.train --seed 13 --fp16                   # full run, about 4-5 min on a T4/L4
python -m strong_baseline.train --seed 42 --fp16                   # repeat for 2026, 31415, 27182
```

`requirements.txt` pins the versions used for the reported results (PyTorch 2.14.1, Transformers 5.22.1).

On Colab, the easiest way is the notebook [`strong_baseline/colab.ipynb`](https://colab.research.google.com/github/3XIA00/CIS5300_Final_Project/blob/m2-strong-baseline/strong_baseline/colab.ipynb) (opens directly in Colab; after merging, replace `m2-strong-baseline` in the link with `main`). It checks the GPU, runs the smoke test and the seeds, prints the results table, and saves `output/roberta-base/` by download, Google Drive or a push to the branch. Usually one person collects everyone's zip and pushes; the push option asks for a fine-grained token scoped to this repo and never prints or stores it. To run the commands by hand instead: Runtime → Change runtime type → GPU, then

```bash
!git clone https://github.com/3XIA00/CIS5300_Final_Project.git
%cd CIS5300_Final_Project
!pip install -q -r strong_baseline/requirements.txt
!python -m strong_baseline.train --seed 13 --fp16
```

Each run writes, per `interface.md`:

- `output/roberta-base/seed<N>/dev.jsonl` and `test.jsonl`: predictions (commit these)
- `output/roberta-base/seed<N>/run.json`: all settings, dev scores per epoch, best epoch, dev and test scores, training time, peak GPU memory, GPU name, token-length statistics, library versions, model revision and the git commit of the code (commit this; runs made before the commit field was added do not have it)
- `checkpoints/roberta-base/seed<N>/model.pt`: best weights, about 500 MB (git-ignored; do not commit)

Scores are printed at the end of training. To recompute them with the shared scorer:

```bash
python score.py --gold data/processed/test.jsonl --pred output/roberta-base/seed13/test.jsonl
python score.py --gold data/processed/test.jsonl --pred output/roberta-base/seed13/test.jsonl \
    --exclude-ids eval/suspected_overlap_test_ids.json
```

To regenerate predictions from a saved checkpoint without retraining:

```bash
python -m strong_baseline.predict --checkpoint checkpoints/roberta-base/seed13 --split test \
    --out output/roberta-base/seed13/test.jsonl
```

Other options: `python -m strong_baseline.train --help` (for example `--train-split train_clean` for the sensitivity check, `--device cpu`, `--no-checkpoint`).

## Results

Five seeds, each trained once on Google Colab with `--fp16` and all other settings as above (same code, `roberta-base` revision `e2da8e2`, PyTorch 2.14.1, Transformers 5.22.1). Scores are weighted F1 (W-F1) and macro F1 (M-F1) on a 0–100 scale, computed by `score.py`; "excl. overlap" leaves out the 11 suspected-overlap test utterances (`eval/suspected_overlap_test_ids.json`). The last row is the mean and sample standard deviation (ddof = 1) over the five seeds. Predictions and full logs are in `output/roberta-base/seed<N>/`.

| Seed | GPU | Dev W-F1 | Dev M-F1 | Test W-F1 | Test M-F1 | Test excl. overlap W-F1 | Test excl. overlap M-F1 | Best epoch | Train time | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | L4 | 59.32 | 45.48 | 63.90 | 47.07 | 63.95 | 47.12 | 4 | 4.8 min | 2.4 GB |
| 42 | L4 | 59.87 | 46.23 | 63.22 | 45.55 | 63.25 | 45.61 | 2 | 3.9 min | 2.4 GB |
| 2026 | L4 | 58.75 | 42.29 | 63.72 | 45.72 | 63.71 | 45.66 | 2 | 3.9 min | 2.4 GB |
| 31415 | T4 | 59.76 | 42.59 | 62.21 | 41.94 | 62.20 | 41.85 | 2 | 3.9 min | 2.4 GB |
| 27182 | T4 | 60.17 | 41.77 | 63.61 | 43.83 | 63.61 | 43.79 | 2 | 3.9 min | 2.4 GB |
| **mean ± sd** | | **59.57 ± 0.55** | **43.67 ± 2.03** | **63.33 ± 0.67** | **44.82 ± 1.98** | **63.34 ± 0.69** | **44.81 ± 2.03** | | | |

Test F1 per class (range over the five seeds):

| neutral | surprise | fear | sadness | joy | disgust | anger |
|---|---|---|---|---|---|---|
| 78.6–80.1 | 55.7–58.4 | 3.6–21.4 | 32.8–36.8 | 57.0–62.3 | 12.2–26.4 | 44.5–49.0 |

Observations:

- The baseline reaches 63.33 test weighted F1, far above the majority-class baseline (31.27; `scoring.md`) and close to the 63.46 that EmoBERTa reports without context using the larger RoBERTa-large (five-seed mean; Kim and Vossen, 2021). Our goal is a controlled comparison between conditions, not matching that number.
- Weighted F1 is stable across seeds (sd 0.67), while macro F1 varies about three times as much (sd ~2.0). The variation comes from the rare classes: `fear` (50 test utterances) and `disgust` (68) are rarely predicted, and their F1 swings between seeds. Comparisons in Milestone 3 should therefore look at macro F1 and these classes and use all seeds, not a single run.
- Excluding the 11 suspected-overlap utterances changes the mean scores by 0.01 (at most 0.09 for a single seed), so the overlap found in Milestone 1 does not affect the conclusions.
- Most runs peak at epoch 2 and stop at epoch 4; dev macro F1 often keeps rising after the weighted-F1 peak, but model selection uses dev weighted F1 as fixed in advance.
- Runs were split across two Colab GPU types (L4 and T4); this can cause tiny numerical differences but does not change the setup.

## References

- Taewoon Kim and Piek Vossen. 2021. [EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa](https://arxiv.org/abs/2108.12009). *arXiv:2108.12009*.
- Yinhan Liu et al. 2019. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). *arXiv:1907.11692*.
- Ilya Loshchilov and Frank Hutter. 2019. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). *ICLR*.
