"""Trim right padding from the sequence tensors after collation."""
import os

from transformers import default_data_collator


def dynamic_padding_enabled():
    return os.environ.get('CISC473_DYNAMIC_PADDING', '1') == '1'


def trim_sequence_padding(batch):
    if dynamic_padding_enabled():
        longest = int(batch['attention_mask'].sum(dim=1).max())
        width = min(batch['attention_mask'].shape[1], (longest + 7) // 8 * 8)
        for key in ('input_ids', 'attention_mask', 'token_type_ids', 'word_ids'):
            if key in batch:
                batch[key] = batch[key][:, :width].contiguous()
    return batch


def collate_bert(examples):
    return trim_sequence_padding(default_data_collator(examples))
