"""RoBERTa-base strong baseline for MELD emotion recognition (Milestone 2, condition B).

Modules:
    data     -- loading records and building model inputs (no torch needed)
    model    -- RoBERTa encoder + linear classifier on the <s> token
    train    -- training, dev-based checkpoint selection, test predictions, scores
    predict  -- regenerate predictions from a saved checkpoint

See strong-baseline.md for how to run it.
"""
