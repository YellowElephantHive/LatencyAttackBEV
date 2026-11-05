import os
import sys
import unittest
from unittest.mock import patch

from bevdet_experiment.cli import parse_args


class CliTests(unittest.TestCase):
    def test_attack_defaults_match_legacy_interface(self):
        old_local_rank = os.environ.pop('LOCAL_RANK', None)
        try:
            with patch.object(sys, 'argv', ['run_bevdet_experiment.py', '--eval', 'bbox']):
                args = parse_args()
            self.assertFalse(args.attack)
            self.assertEqual(args.eps, 8 / 255.0)
            self.assertEqual(args.alpha, 2 / 255.0)
            self.assertEqual(args.steps, 5)
            self.assertEqual(args.hm_loss_mode, 'gate_sum')
            self.assertEqual(args.attack_cams, 'all')
            self.assertEqual(os.environ['LOCAL_RANK'], '0')
        finally:
            if old_local_rank is None:
                os.environ.pop('LOCAL_RANK', None)
            else:
                os.environ['LOCAL_RANK'] = old_local_rank

    def test_legacy_options_conflict_is_rejected(self):
        argv = [
            'run_bevdet_experiment.py', '--eval', 'bbox',
            '--options', 'a=1', '--eval-options', 'b=2',
        ]
        with patch.object(sys, 'argv', argv):
            with self.assertRaises(ValueError):
                parse_args()


if __name__ == '__main__':
    unittest.main()
