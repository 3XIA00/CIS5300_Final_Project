# Emotion Recognition in Conversation with Emoji-Aware Representations

**CIS 5300 - Milestone 1: Literature Review and Data**

We study text-only emotion recognition in conversation (ERC): predict one of seven emotions for each utterance using text and speaker identities. Our question is whether frozen emoji-pretrained features add value beyond a fine-tuned RoBERTa classifier and dialogue history. The following three papers motivate the design; all reported scores are published results, not our experiments.

## 1. Literature Review

### 1.1 DeepMoji: Transferring emotion-sensitive representations

Felbo et al. [1] pretrain DeepMoji to predict 64 emoji categories using approximately **1.246 billion tweets**. Two bidirectional LSTM layers, attention, and skip connections produce transferable representations. They compare training from scratch, frozen-feature classification, full fine-tuning, and *chain-thaw*, which tunes layers individually before jointly fine-tuning. Emoji are noisy supervision rather than exact emotion labels, and transfer does not require emoji in downstream inputs.

Across eight emotion, sentiment, and sarcasm benchmarks, the method improves over the comparison systems. On the seven-class PsychExp emotion dataset, **macro F1 is 0.57 with chain-thaw and 0.56 with frozen features**, versus **0.45** for the comparison baseline (Table 5; five-run averages). The frozen-feature result supports our feature-fusion extension. However, MELD is not evaluated: transfer from tweets to television dialogue remains an empirical question. A random frozen counterpart will control architecture and feature count, while tweet-domain exposure remains a separate potential explanation. [1]

### 1.2 DialogueRNN: Tracking speakers and history

Majumder et al. [2] model conversational history, participant states, and emotion representations with three interacting GRUs. Attention selects relevant previous global states when updating a speaker, distinguishing shared dialogue context from each participant's evolving state. Their text-only system begins with CNN utterance features. This architecture motivates preserving speaker identity when constructing our context inputs.

On IEMOCAP, the best text-only variant, BiDialogueRNN with attention, obtains **62.75 weighted F1**, versus **56.13** for the conversational memory network baseline (Table 2). Removing party states lowers vanilla DialogueRNN from 59.89 to 55.56 F1 (Table 4), supporting speaker modeling. The original experiments use IEMOCAP and AVEC, not MELD; moreover, the strongest variant uses future utterances. We therefore use its findings as motivation rather than an exactly matched baseline, and begin with simpler speaker-marked history concatenation before considering recurrent models. [2]

### 1.3 EmoBERTa: A directly relevant RoBERTa baseline

Kim and Vossen [3] prepend speaker names to utterances, separate context and target segments with special tokens, and fine-tune **RoBERTa-large** end to end with a linear classification head. This approach models speaker interactions through the pretrained Transformer without an additional dialogue RNN or graph. It directly supports both our target-only strong baseline and our initial concatenated-context extension.

On MELD, weighted F1 rises from **63.46 without context** to **64.55 with past-only context** and **65.61 with past and future context** (Table 2; five-seed means). These ablations closely match our research design. We will use **RoBERTa-base**, past-only inputs, and the same concatenation architecture in every context-enabled condition. Backbone size, context limits, and our cleaned training copy differ from the paper, so this is a scoped adaptation rather than a claim to reproduce its scores. EmoBERTa also establishes that concatenation itself is prior work; our contribution is the controlled feature-transfer analysis. [3]

## 2. Collected Data

We collected official **MELD** text-and-annotation CSVs [4] at revision `e8cedf27b5d2877e198332c957127e16eb214afe`, preserving the raw files and creating sorted UTF-8 JSONL records. Labels are anger, disgust, fear, joy, neutral, sadness, and surprise. Speaker, dialogue ID, and utterance order support context construction; audio, video, and gold sentiment/emotion labels are excluded from model inputs.

| Split | Utterances | Observed dialogue IDs |
|---|---:|---:|
| Official train | 9,989 | 1,038 |
| Clean train for our experiments | 9,988 | 1,038 |
| Development | 1,109 | 114 |
| Test | 2,610 | 280 |

These are measured counts; training dialogue ID 60 is absent, explaining the repository description's larger dialogue count. A real training example is `(train:0:2, Chandler, "That I did. That I did.") -> neutral`.

Checks found no duplicate within-split utterance IDs, empty texts/speakers, or unknown labels. One identical source-time occurrence spans `train:12:7` and `test:85:0`; clean training excludes it from both targets and history. Dev/test remain unchanged, repeated conversational strings are retained, and original ID gaps are preserved. Neutral represents **47.15%** of clean training data; fear and disgust each represent about **2.7%**. Full provenance, class counts, checksums, and cleaning details accompany the data archive in `data.md`.

## 3. Mentor-Informed Experimental Plan

Following Ethan's feedback, we will **fine-tune RoBERTa-base in all six neural conditions**: history off/on crossed with no auxiliary features, frozen random DeepMoji features, or frozen pretrained DeepMoji features. Context-enabled conditions share one history-concatenation architecture. Random and pretrained branches share architecture, tokenizer, output dimension, preprocessing, and classifier size. Shared training settings and at least **three matched seeds** will produce weighted and macro F1 **mean +/- sample standard deviation**; five seeds and dialogue-level paired bootstrap intervals are planned if resources permit. Configuration selection uses dev; final evaluation uses test.

Implementation will first establish MELD loading, the RoBERTa baseline, and scoring, then verify TorchMoji loading and feature extraction before adding extensions. Null or negative results will be analyzed for domain mismatch, redundant information, or optimization effects. These are planned analyses, not completed findings; details and controls are specified in `experiment_plan.md`.

## References

[1] Bjarke Felbo, Alan Mislove, Anders Søgaard, Iyad Rahwan, and Sune Lehmann. 2017. [Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://aclanthology.org/D17-1169/). *EMNLP*, 1615-1625.

[2] Navonil Majumder, Soujanya Poria, Devamanyu Hazarika, Rada Mihalcea, Alexander Gelbukh, and Erik Cambria. 2019. [DialogueRNN: An Attentive RNN for Emotion Detection in Conversations](https://arxiv.org/abs/1811.00405v4). *AAAI*, 33(01):6818-6825.

[3] Taewoon Kim and Piek Vossen. 2021. [EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa](https://arxiv.org/abs/2108.12009). *arXiv:2108.12009*.

[4] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. [MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/). *ACL*, 527-536. [Official data repository](https://github.com/declare-lab/MELD).
