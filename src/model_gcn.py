"""Aspect-aware dependency GCN on BERT subword embeddings."""
import torch
from torch import nn
from transformers import AutoModel


class BertDependencyGCN(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=3, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.gcn1 = nn.Linear(hidden, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                word_ids=None, adjacency=None, aspect_mask=None, labels=None):
        bert_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            bert_kwargs['token_type_ids'] = token_type_ids
        hidden = self.bert(**bert_kwargs).last_hidden_state
        cls = hidden[:, 0]
        b, seq_len, d = hidden.shape
        n = adjacency.shape[-1]
        # Group first-sequence BERT subwords by their spaCy word index.
        safe_idx = word_ids.clamp(min=0)
        valid = (word_ids >= 0).unsqueeze(-1).to(hidden.dtype)
        words = hidden.new_zeros((b, n, d))
        counts = hidden.new_zeros((b, n, 1))
        words.scatter_add_(1, safe_idx.unsqueeze(-1).expand(-1, -1, d), hidden * valid)
        counts.scatter_add_(1, safe_idx.unsqueeze(-1), valid)
        words = words / counts.clamp(min=1.0)
        x = self.dropout(self.activation(self.gcn1(torch.bmm(adjacency, words))))
        x = self.dropout(self.activation(self.gcn2(torch.bmm(adjacency, x))))
        mask = aspect_mask.unsqueeze(-1).to(x.dtype)
        aspect = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(self.dropout(torch.cat([cls, aspect], dim=-1)))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {'loss': loss, 'logits': logits}
