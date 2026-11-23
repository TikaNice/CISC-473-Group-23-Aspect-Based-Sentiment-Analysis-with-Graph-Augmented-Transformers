import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import run_one
from src.result_paths import result_path


class RunOneResumeTests(unittest.TestCase):
    def test_validated_results_are_skipped_with_training_tags(self):
        for graph, tag in [('none', 'aspectfix'), ('dependency', 'aspectfix'),
                           ('identity', 'identity'), ('full', 'fullgraph')]:
            with self.subTest(graph=graph), TemporaryDirectory() as tmp:
                result = Path(tmp) / result_path('gcn', 'restaurant', 0.1, 42, tag)
                result.parent.mkdir(parents=True, exist_ok=True)
                result.touch()
                with patch.object(sys, 'argv', ['run_one', 'gcn', 'restaurant', '0.1', '42', graph]), \
                     patch.object(run_one, 'result_path', return_value=result), \
                     patch.object(run_one, 'validate_result') as validate, \
                     patch.object(run_one.subprocess, 'call') as call:
                    self.assertEqual(run_one.main(), 0)
                    run_one.result_path.assert_called_once_with('gcn', 'restaurant', 0.1, 42, tag)
                    validate.assert_called_once_with('gcn', 'restaurant', 0.1, 42,
                                                     'dependency' if graph == 'none' else graph)
                    call.assert_not_called()

    def test_invalid_evidence_restarts_the_run(self):
        with TemporaryDirectory() as tmp:
            result = Path(tmp) / 'broken.json'
            result.write_text('{"partial":', encoding='utf-8')
            with patch.object(sys, 'argv', ['run_one', 'bert', 'restaurant', '0.1', '42', 'none']), \
                 patch.object(run_one, 'result_path', return_value=result), \
                 patch.object(run_one, 'validate_result', side_effect=ValueError('invalid JSON')), \
                 patch.object(run_one.subprocess, 'call', return_value=0) as call:
                self.assertEqual(run_one.main(), 0)
                call.assert_called_once()


if __name__ == '__main__':
    unittest.main()
