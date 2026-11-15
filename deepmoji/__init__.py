"""Frozen DeepMoji (torchMoji) features for MELD: setup, extraction and feasibility checks.

Modules:
    setup_torchmoji   -- clone pinned torchMoji, apply the Python 3 / PyTorch 2 fixes, download weights
    common            -- shared paths and torchMoji loading (adds third_party/torchMoji to sys.path)
    check_torchmoji   -- reproduce torchMoji's own reference outputs; determinism and freezing checks
    extract_features  -- 2304-dim features for every MELD utterance, plus unknown-word statistics
    probe             -- logistic regression on DeepMoji features only (is there emotion signal?)
    fusion_smoke      -- RoBERTa-base + DeepMoji concatenation runs end to end (64 utterances)

See deepmoji.md for how to run it and what the checks found.
"""
