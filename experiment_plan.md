# Controlled Experiment Plan After the Mentor Meeting

This working plan combines `../milestone1_papers.md` with the user-provided meeting summary from Ethan. It specifies the next implementation stage; it does not claim that models have been trained or TorchMoji compatibility has been verified. The checkpoint, seed list, and numerical defaults below are project working choices that operationalize the feedback, rather than exact settings prescribed by Ethan.

## 1. Research questions and literature alignment

1. Does speaker-marked dialogue history improve a target-only text classifier?
2. Do frozen pretrained DeepMoji features improve the same classifier, with and without history?
3. Does that improvement exceed what an equally sized, architecturally identical frozen random encoder provides?

The three core reviews are **DeepMoji, DialogueRNN, and EmoBERTa**. EmoBERTa replaces DialogueGCN as a full review because its input-concatenation design directly matches the initial implementation. DialogueGCN remains useful background, rather than an additional required implementation.

EmoBERTa already establishes speaker-aware concatenation as prior work. Our proposed contribution is the controlled evaluation of added features and their interaction with that context method, not a claim to invent concatenation.

## 2. Shared context architecture and six conditions

Use **RoBERTa-base, fine-tuned end to end**, plus a linear seven-way classifier on the final-layer `<s>` hidden state. Use this pooling choice in every condition. This is an adaptation of the EmoBERTa baseline family; its paper uses RoBERTa-large and also studies future context. We use past context only and do not treat its published numbers as matched reproduction targets.

For every target utterance, prepend its speaker name and retain its text. In history-enabled conditions, prepend up to N previous retained turns of the same dialogue, with speaker names and RoBERTa separators. Clearly separate the history segment from the target segment. The classification target is always the current utterance, never a pooled dialogue label.

| ID | Dialogue history | Auxiliary encoder | RoBERTa | Auxiliary weights |
|---|---|---|---|---|
| B | None | None | Fine-tuned | Not applicable |
| C | Concatenated history | None | Fine-tuned | Not applicable |
| R | None | Random DeepMoji architecture | Fine-tuned | Frozen |
| D | None | Pretrained DeepMoji | Fine-tuned | Frozen |
| CR | Same history as C | Random DeepMoji architecture | Fine-tuned | Frozen |
| CD | Same history as C | Pretrained DeepMoji | Fine-tuned | Frozen |

The deterministic majority-class baseline is reported separately. There are six neural conditions, so the three-seed main comparison requires **18 training runs**, excluding development searches and debugging runs. Expanding every condition to five seeds requires 30 runs.

The C, CR, and CD conditions must use exactly the same context builder, N, speaker formatting, maximum input length, truncation rule, pooling, and classification target. Do not compare concatenation in C against a GRU or Transformer in CD. A later hierarchical architecture would require a separate matched comparison and is optional.

Initial context choices: maximum 256 RoBERTa tokens; choose N from {1, 3, 5} using development performance of the context-only model, then lock it for C, CR, and CD. Preserve the complete target when possible; remove oldest history first. If the target alone exceeds the limit, apply the same documented target truncation to every condition. Context windows use observed sorted turns, not arithmetic on potentially missing utterance IDs.

## 3. Matched pretrained and random feature branches

DeepMoji processes only the current utterance in all feature-enabled conditions. Extract its **2304-dimensional attention representation**, not the 64-way emoji-probability vector.

For the random control, instantiate the same feature-extraction architecture with random parameters, including the word embeddings, recurrent layers, and attention parameters. Do not retain pretrained embeddings in this control. Keep vocabulary/tokenizer, sequence-length policy, parameter shapes, output dimension, padding conventions, and numerical feature processing identical to the pretrained branch.

Both branches have `requires_grad=False`, run in evaluation mode, and are evaluated without gradients. Cache frozen outputs; use a separate random encoder initialization for each training seed. The pretrained weights remain identical across seeds. If auxiliary features are standardized, use the same per-dimension procedure in both branches, fit means/scales on clean training records only, and reuse those statistics for dev/test.

Concatenate the auxiliary vector with RoBERTa's target-input representation before classification. D and R, and CD and CR, therefore have identical classifier shapes and trainable parameter counts. Feature-free models have a smaller classifier; we do not claim that concatenation keeps that classifier fixed.

Interpret the contrasts separately:

- **C minus B:** contribution of the chosen history method.
- **D minus B / CD minus C:** incremental utility of adding pretrained features.
- **D minus R / CD minus CR:** utility of pretrained versus random features under matched auxiliary architecture and dimensionality.

The random control helps rule out feature count alone. It does **not** uniquely attribute a gain to emoji supervision: pretrained DeepMoji also contains learned lexical knowledge and exposure to Twitter. A future BERTweet comparison could probe domain transfer, but different architectures, tokenizers, and corpora prevent it from being a clean causal isolation of the emoji objective.

## 4. Training consistency and reporting

Use the same three seeds, **13, 42, and 2026**, for all six conditions. If the budget permits five, add 31415 and 27182 for every condition, rather than only favorable models. Separate random-number streams for auxiliary initialization from RoBERTa/head initialization and data shuffling, so constructing the random encoder does not unintentionally change shared initialization.

Working defaults are AdamW, learning rate 2e-5, effective batch size 16, at most five epochs, weight decay 0.01, and linear decay after 10% warmup. Use ordinary unweighted cross entropy in all conditions. Evaluate at each epoch end; select the best checkpoint by dev weighted F1 and stop after two consecutive epochs without improvement, retaining the earlier checkpoint on ties. Lock any revised settings from dev before the final comparison; do not perform unequal per-condition searches or change RoBERTa from fine-tuned to frozen between arms.

Keep the clean train/dev/test protocol from `data.md` fixed. Store predictions with sample IDs for each condition/seed. Report weighted F1 and macro F1 on the same seven labels, plus per-class scores and confusion matrices. Define F1 for an unpredicted class as zero, and use a consistent 0-100 F1 scale in result tables. Report the arithmetic mean and **sample standard deviation (ddof=1)** across seeds, not the best seed or standard error.

Paired bootstrap is an optional uncertainty check. Resample **whole test dialogues**, with replacement, using the identical resampled IDs for both models and every matched seed. Include all turns from each sampled dialogue, recompute both metrics, and average paired score differences across seeds. Use 2,000 resamples to obtain percentile 95% intervals. This respects within-dialogue dependence; the interval describes test-dialogue sampling uncertainty conditional on the trained runs, while across-seed standard deviations describe training variability. Test labels are used only for the final fixed evaluation and subsequent descriptive error analysis, not configuration selection.

## 5. Implementation order and interpretation

1. Establish MELD loading, chronological context boundaries, label mapping, majority baseline, and a common scoring pipeline. Train one RoBERTa baseline run to validate the complete path from inputs to saved predictions and scores.
2. During baseline setup, test TorchMoji weight/vocabulary loading and feature extraction on approximately ten utterances. Check shape `(number_of_utterances, 2304)`, finite values, deterministic evaluation outputs, and freezing. Repeat the shape/freezing checks for the random counterpart. This is an early compatibility check, not a completed milestone claim.
3. Validate the shared concatenation builder, cache features with checkpoint/tokenizer/seed identifiers, and lock the development choices. Then execute the matched six-condition, three-seed comparison.
4. If features help, compare pretrained against random controls and inspect which classes or cases improve. If they do not, investigate redundancy with context, source-domain mismatch, feature scaling, truncation, or optimization as hypotheses. Distinguish a near-zero estimate with high variability from consistent degradation. A non-improvement is a valid research result, not a reason to select favorable seeds or invent a mechanism.

## 6. Checked literature and supplementary context

- **EmoBERTa:** Kim and Vossen (2021), [arXiv:2108.12009](https://arxiv.org/abs/2108.12009). Table 2 reports MELD weighted F1 of 63.46 without context, 64.55 with past-only context, and 65.61 with past plus future context, averaged over five seeds. These are RoBERTa-large results. Appendix A.1 confirms no weights are frozen during training.
- **DialogueRNN:** Majumder et al. (2019), [original paper](https://arxiv.org/abs/1811.00405v4). Original evaluation is IEMOCAP/AVEC; a MELD number cited by later work should not be misattributed to that original experiment.
- **DeepMoji:** Felbo et al. (2017), [original paper](https://aclanthology.org/D17-1169/). The paper evaluates frozen features as well as chain-thaw and full fine-tuning, so our frozen branch has a direct methodological precedent.
- **RoBERTa:** Liu et al. (2019), [model paper](https://arxiv.org/abs/1907.11692). Record the exact checkpoint revision, tokenizer, and software versions when the pipeline is implemented.
- **COSMIC:** Ghosal et al. (2020), [Findings of EMNLP](https://aclanthology.org/2020.findings-emnlp.224/). It combines RoBERTa utterance features and COMET-derived commonsense representations through recurrent state modeling. It motivates external information in ERC, but is not the same architecture as our direct concatenation classifier.
- **DialogueGCN:** Ghosal et al. (2019), [EMNLP-IJCNLP](https://aclanthology.org/D19-1015/). Speaker/temporal relation graphs are useful context-modeling background; implementing them is outside the initial scope.
- **BERTweet:** Nguyen, Vu, and Nguyen (2020), [EMNLP System Demonstrations](https://aclanthology.org/2020.emnlp-demos.2/). Its masked-language-model objective differs from emoji classification. It can be an optional domain-transfer probe, not evidence that it has learned nothing about emojis.

We keep optional references and implementation details in this supplementary plan so the M1 writeup can retain three focused two-paragraph reviews and a concise data description.
