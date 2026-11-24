import os
import unittest
from unittest.mock import patch

import torch

from src.batch_padding import trim_sequence_padding


class BatchPaddingTests(unittest.TestCase):
    def test_trims_all_sequence_fields_and_keeps_graph_and_labels(self):
        batch = {
            'input_ids': torch.arange(32).reshape(2, 16),
            'attention_mask': torch.tensor([[1] * 5 + [0] * 11, [1] * 7 + [0] * 9]),
            'token_type_ids': torch.zeros(2, 16),
            'word_ids': torch.tensor([[0] * 5 + [-1] * 11, [0] * 7 + [-1] * 9]),
            'adjacency': torch.ones(2, 3, 3),
            'aspect_mask': torch.ones(2, 3),
            'labels': torch.tensor([0, 1]),
        }
        original = batch.copy()
        with patch.dict(os.environ, {'CISC473_DYNAMIC_PADDING': '1'}):
            trimmed = trim_sequence_padding(batch)
        for key in ('input_ids', 'attention_mask', 'token_type_ids', 'word_ids'):
            torch.testing.assert_close(trimmed[key], original[key][:, :8])
        for key in ('adjacency', 'aspect_mask', 'labels'):
            self.assertIs(trimmed[key], original[key])
        with patch.dict(os.environ, {'CISC473_DYNAMIC_PADDING': '0'}):
            self.assertEqual(trim_sequence_padding(original)['input_ids'].shape[1], 16)
