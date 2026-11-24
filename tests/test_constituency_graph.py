import unittest
from types import SimpleNamespace

import numpy as np

from src.constituency_graph import adjacency, phrase_spans


def node(label, *children):
    return SimpleNamespace(label=label, children=children)


class ConstituencyGraphTest(unittest.TestCase):
    def test_phrase_edges_and_truncation(self):
        tree = node('ROOT', node('S',
                    node('NP', node('The'), node('food')),
                    node('VP', node('was'), node('very'), node('good'))))
        spans = phrase_spans(tree)
        self.assertEqual(spans, [(0, 2), (2, 5)])
        expected = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
        np.testing.assert_array_equal(adjacency(spans, {0: 0, 1: 1, 2: 2, 4: 3}), expected)
