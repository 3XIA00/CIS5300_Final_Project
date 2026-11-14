"""RoBERTa encoder with a linear classifier on the final-layer <s> hidden state.

    h = RoBERTa(x)[<s>]            (768-dim for roberta-base)
    p = softmax(W [h; a] + b)      a = optional auxiliary features (DeepMoji in M3)

For the Milestone 2 baseline aux_dim = 0, so p = softmax(W h + b).
"""

import torch
from torch import nn
from transformers import AutoModel


class UtteranceClassifier(nn.Module):
    def __init__(self, model_name, num_labels, aux_dim=0, dropout=0.1):
        super().__init__()
        # The pooler is not used: classification reads the raw <s> state (experiment_plan.md).
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.aux_dim = aux_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + aux_dim, num_labels)

    def forward(self, input_ids, attention_mask, aux_features=None):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = states[:, 0]  # <s> token
        if self.aux_dim:
            if aux_features is None:
                raise ValueError(f"model expects {self.aux_dim}-dim aux_features")
            h = torch.cat([h, aux_features], dim=-1)
        elif aux_features is not None:
            raise ValueError("model was built with aux_dim=0 but aux_features were given")
        return self.classifier(self.dropout(h))
