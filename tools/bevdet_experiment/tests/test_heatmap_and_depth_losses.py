import math
import unittest

import torch

from bevdet_experiment.losses.depth import (
    _depth_prob_from_depthnet_out,
    depth_expected_bin_and_entropy,
    depth_loss_entropy_from_x,
    depth_loss_logp_fixed_from_x,
)
from bevdet_experiment.losses.heatmap import (
    hm_loss_gate_sum,
    hm_loss_overload_hardish,
    hm_loss_peak_count,
    hm_loss_topk_sum,
)


class HeatmapLossTests(unittest.TestCase):
    def test_empty_tasks_return_zero(self):
        value = hm_loss_gate_sum({}, tau=0.1, steep=20, device='cpu')
        self.assertEqual(float(value), 0.0)

    def test_gate_sum_matches_direct_formula(self):
        logits = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)
        actual = hm_loss_gate_sum({0: logits}, tau=0.2, steep=3.0, device='cpu')
        probability = torch.sigmoid(logits)
        expected = torch.sigmoid(3.0 * (probability - 0.2)).sum()
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_heatmap_modes_return_finite_scalars(self):
        logits = torch.tensor([[[[0.0, 1.0], [-1.0, 2.0]]]], requires_grad=True)
        values = [
            hm_loss_overload_hardish({0: logits}, 0.1, 20, 'cpu'),
            hm_loss_peak_count({0: logits}, 0.1, 20, 3, 40, 1e-3, 'cpu'),
            hm_loss_topk_sum({0: logits}, 200, 'cpu'),
        ]
        for value in values:
            self.assertEqual(value.ndim, 0)
            self.assertTrue(bool(torch.isfinite(value)))


class DepthLossTests(unittest.TestCase):
    def test_depth_probability_uses_first_d_channels(self):
        logits = torch.zeros((1, 5, 1, 1))
        probability = _depth_prob_from_depthnet_out(logits, D=3)
        self.assertEqual(tuple(probability.shape), (1, 3, 1, 1))
        self.assertTrue(torch.allclose(
            probability.sum(dim=1), torch.ones((1, 1, 1))))

    def test_uniform_depth_statistics(self):
        probability = torch.full((1, 3, 1, 1), 1 / 3)
        expected_bin, entropy = depth_expected_bin_and_entropy(probability)
        self.assertTrue(torch.allclose(expected_bin, torch.ones((1, 1, 1))))
        self.assertTrue(torch.allclose(
            entropy, torch.full((1, 1, 1), math.log(3))))

    def test_uniform_depth_losses(self):
        logits = torch.zeros((1, 3, 1, 1), requires_grad=True)
        entropy = depth_loss_entropy_from_x(logits, D=3)
        logp = depth_loss_logp_fixed_from_x(logits, D=3, kstar=-1)
        self.assertTrue(torch.allclose(entropy, torch.tensor(math.log(3))))
        self.assertTrue(torch.allclose(logp, torch.tensor(-math.log(3))))
        (entropy + logp).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
