from types import SimpleNamespace
import unittest

import torch

from bevdet_experiment.losses.geometry import (
    _parse_three_ints,
    _parse_two_ints,
    geom_area_loss_ascent,
    geom_reg_dispersion_ascent,
    geom_rot_dispersion_ascent,
    geom_volume_loss_ascent,
)
from bevdet_experiment.losses.proposal import (
    decode_topk_props_simple,
    gather_topk_proposals_from_task,
    proposal_diversity_loss_ascent,
    proposal_shape_range_loss_ascent,
    proposal_vertical_loss_ascent,
)


class GeometryLossTests(unittest.TestCase):
    def test_dimension_index_parsing(self):
        self.assertEqual(_parse_two_ints('2,0'), (2, 0))
        self.assertEqual(_parse_three_ints('2,0,1'), (2, 0, 1))
        self.assertEqual(_parse_two_ints('bad'), (0, 1))

    def test_geometry_losses_are_finite_and_differentiable(self):
        dim = torch.zeros((1, 3, 2, 2), requires_grad=True)
        reg = torch.tensor([[[[0.0, 2.0]], [[0.0, 0.0]]]], requires_grad=True)
        rot = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
        gate_dim = torch.ones((1, 1, 2, 2))
        gate_line = torch.ones((1, 1, 1, 2))
        values = [
            geom_area_loss_ascent(dim, gate_dim, '0,1', 1e-6),
            geom_volume_loss_ascent(dim, gate_dim, '0,1,2', 1e-6),
            geom_reg_dispersion_ascent(reg, gate_line, 1e-6),
            geom_rot_dispersion_ascent(rot, gate_line, 1e-6),
        ]
        for value in values:
            self.assertTrue(bool(torch.isfinite(value)))
        sum(values).backward()
        self.assertIsNotNone(dim.grad)
        self.assertIsNotNone(reg.grad)
        self.assertIsNotNone(rot.grad)


class ProposalLossTests(unittest.TestCase):
    def test_topk_gather_and_decode_shapes(self):
        heatmap = torch.tensor([[[[0.0, 3.0], [1.0, 2.0]]]])
        outputs = {
            'reg': torch.zeros((1, 2, 2, 2)),
            'height': torch.zeros((1, 1, 2, 2)),
            'dim': torch.zeros((1, 3, 2, 2)),
            'rot': torch.zeros((1, 2, 2, 2)),
        }
        proposals = gather_topk_proposals_from_task(outputs, heatmap, topk=2)
        self.assertEqual(tuple(proposals['score'].shape), (1, 2))
        boxes = decode_topk_props_simple(proposals, norm_bbox=True)
        self.assertEqual(tuple(boxes['cx'].shape), (1, 2))
        self.assertTrue(torch.allclose(boxes['l'], torch.ones((1, 2))))
        self.assertTrue(torch.allclose(boxes['w'], torch.ones((1, 2))))
        self.assertTrue(torch.allclose(boxes['h'], torch.ones((1, 2))))

    def test_proposal_regularizers_are_finite(self):
        boxes = {
            'l': torch.tensor([[0.5, 6.0]]),
            'w': torch.tensor([[0.2, 4.0]]),
            'h': torch.tensor([[0.2, 4.0]]),
            'z': torch.tensor([[-3.0, 5.0]]),
            'cx': torch.tensor([[0.0, 0.5]]),
            'cy': torch.tensor([[0.0, 0.0]]),
        }
        args = SimpleNamespace(
            shape_lmin=1.0, shape_lmax=5.0,
            shape_wmin=0.5, shape_wmax=3.0,
            shape_hmin=0.5, shape_hmax=3.0,
            vert_zmin=-2.0, vert_zmax=4.0,
            div_sample_k=2, div_min_dist=1.5,
        )
        values = [
            proposal_shape_range_loss_ascent(boxes, args),
            proposal_vertical_loss_ascent(boxes, args),
            proposal_diversity_loss_ascent(boxes, args),
        ]
        for value in values:
            self.assertTrue(bool(torch.isfinite(value)))
            self.assertLessEqual(float(value), 0.0)


if __name__ == '__main__':
    unittest.main()
