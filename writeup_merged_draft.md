# Emotion Recognition in Conversation with Emoji-Aware Representations

**CIS 5300 — Milestone 1: Literature Review and Data**

We study text-only emotion recognition in conversation (ERC): predicting one of seven emotions for each utterance using text and speaker identities. Our question is whether frozen emoji-pretrained features add information beyond a fine-tuned RoBERTa classifier and dialogue history. All scores below are published results, not our experiments.

## 1. Literature Review

### 1.1 DeepMoji: Emoji-supervised representation learning

Felbo et al. [1] pretrain DeepMoji to predict 64 emoji categories from approximately 1.246 billion tweets. Two bidirectional LSTM layers, attention, and skip connections produce transferable text representations. Emoji provide noisy distant supervision rather than exact emotion labels. The paper compares training from scratch, frozen-feature classification, full fine-tuning, and chain-thaw, which tunes individual layers before jointly fine-tuning. Downstream inputs do not need to contain emoji.

Across eight emotion, sentiment, and sarcasm benchmarks, DeepMoji improves over the comparison systems. On seven-class PsychExp, macro F1 is 0.57 with chain-thaw and 0.56 with frozen features, versus 0.45 for the comparison baseline (Table 5; five-run averages). The frozen-feature result supports our feature-fusion extension. However, the paper does not evaluate MELD, leaving transfer from tweets to television dialogue an empirical question. [1]

### 1.2 EmoBERTa: Speaker-aware RoBERTa with context

Kim and Vossen [2] prepend speaker names to utterances, separate context and target segments with special tokens, and fine-tune RoBERTa-large end to end with a linear classifier. The pretrained Transformer models speaker interactions without an additional dialogue RNN or graph. This provides a direct starting point for our target-only baseline and speaker-marked history concatenation.

On MELD, weighted F1 rises from 63.46 without context to 64.55 with past-only context and 65.61 with past and future context (Table 2; five-seed means). We will use RoBERTa-base, past-only inputs, and identical concatenation in all context-enabled conditions. Different backbone size, context limits, and training settings make this a scoped adaptation rather than an exact reproduction. Concatenation itself is established prior work; our contribution is the controlled evaluation of additional emoji-pretrained features. [2]

### 1.3 CoMPM: Context and pretrained speaker memory

Lee and Lee [3] combine a RoBERTa-based context module (CoM) with a pretrained memory module (PM) encoding the current speaker's previous utterances. A GRU tracks these speaker-specific representations, and its output is added to the context representation for classification. This tests whether an additional source of pretrained information improves ERC; our simpler auxiliary branch instead encodes the current utterance with DeepMoji and concatenates its representation.

On MELD, CoMPM achieves 66.52 weighted F1, compared with 64.90 for CoM alone; freezing the pretrained PM encoder yields 65.77 (Table 2; three-run means). The main model fine-tunes both encoders. Its randomly initialized PM variant is trained, whereas our planned random DeepMoji encoder remains frozen. This motivates matching auxiliary architecture and feature count, but a gain over random features would not isolate emoji supervision from other benefits of pretraining, including tweet-domain exposure. [3]

## 2. Collected Data

We collected the official MELD [4] text-and-annotation CSVs at revision `e8cedf27b5d2877e198332c957127e16eb214afe`, preserving raw files and creating sorted UTF-8 JSONL records. Labels are anger, disgust, fear, joy, neutral, sadness, and surprise. Text, speaker, dialogue ID, and utterance order support model inputs; gold emotion and sentiment labels are excluded from inputs. Audio and video are outside our scope.

| Collected split/version | Utterances | Observed dialogue IDs |
|---|---:|---:|
| Official train | 9,989 | 1,038 |
| Additional exact-key-filtered train | 9,988 | 1,038 |
| Development | 1,109 | 114 |
| Test | 2,610 | 280 |

These are measured counts; training dialogue ID 60 is absent. A real example is `(train:0:2, Chandler, "That I did. That I did.") → neutral`. Checks found no duplicate within-split utterance IDs, empty text/speaker fields, or unknown labels. Neutral accounts for approximately 47% of training examples; fear and disgust each account for approximately 2.7%, motivating both weighted and macro F1.

For benchmark comparability, the proposed main protocol retains the official splits. We also supply a restricted training copy removing one exact `(season, episode, start, end, speaker, text)` match, `train:12:7` / `test:85:0`, from targets and history. Further checks identified likely source overlap between `train:12:4–6` and `test:84:0–2`, despite inconsistent episode metadata and emotion labels. The restricted filter therefore does not establish that all source overlap has been eliminated. Dev/test remain unchanged; provenance, schema, examples, and audit details are documented separately in `data.md`.

Following mentor feedback, planned comparisons cross history off/on with no auxiliary features, frozen random DeepMoji, or frozen pretrained DeepMoji. All six conditions will fine-tune RoBERTa-base consistently and report weighted/macro F1 mean ± standard deviation over at least three matched seeds; no training results are claimed here.

## References

[1] Felbo, B., et al. 2017. [Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://aclanthology.org/D17-1169/). *EMNLP*, 1615–1625.

[2] Kim, T., and Vossen, P. 2021. [EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa](https://arxiv.org/abs/2108.12009). *arXiv:2108.12009*.

[3] Lee, J., and Lee, W. 2022. [CoMPM: Context Modeling with Speaker’s Pre-trained Memory Tracking for Emotion Recognition in Conversation](https://aclanthology.org/2022.naacl-main.416/). *NAACL-HLT*, 5669–5679.

[4] Poria, S., et al. 2019. [MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/). *ACL*, 527–536. [Official dataset](https://github.com/declare-lab/MELD).
