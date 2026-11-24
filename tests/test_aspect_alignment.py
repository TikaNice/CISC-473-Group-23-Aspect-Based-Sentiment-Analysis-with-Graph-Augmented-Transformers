"""Regression coverage for case-sensitive SetFit occurrence indices and XML offsets."""
import unittest
from src.train_gcn import find_aspect


class AspectAlignmentTests(unittest.TestCase):
    def test_substring_in_differently_cased_word_is_not_selected(self):
        text = 'Bartender was unable to tear himself away from friends at bar.'
        start, end = find_aspect(text, 'bar', {'ordinal': 0})
        self.assertEqual((start, end), (text.rindex('bar'), text.rindex('bar') + 3))

    def test_lowercase_aspect_does_not_count_capitalized_occurrence(self):
        text = 'Pizza - the only pizza in NYC.'
        self.assertEqual(find_aspect(text, 'pizza', {'ordinal': 0}), (17, 22))

    def test_ordinal_preserves_case_and_does_not_match_inside_those(self):
        text = 'The OS is useful for those switching to this OS.'
        start = text.rindex('OS')
        self.assertEqual(find_aspect(text, 'OS', {'ordinal': 1}), (start, start + 2))

    def test_xml_offsets_take_precedence(self):
        self.assertEqual(find_aspect('pizza pizza', 'pizza', {'start': 6, 'end': 11, 'ordinal': 0}), (6, 11))

    def test_invalid_xml_offsets_are_rejected(self):
        with self.assertRaises(ValueError):
            find_aspect('pizza pizza', 'pizza', {'start': -5, 'end': 11})

    def test_invalid_occurrence_is_rejected(self):
        with self.assertRaises(ValueError):
            find_aspect('Pizza', 'pizza', {'ordinal': 0})


if __name__ == '__main__':
    unittest.main()
