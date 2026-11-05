import unittest

import torch
import torch.nn as nn

from bevdet_experiment.hooks import (
    _get_D_from_model,
    disable_checkpointing,
    register_all_task_heatmap_logits_hooks,
)
from bevdet_experiment.roi import build_single_patch_mask, parse_cam_ids


class CameraAndRoiTests(unittest.TestCase):
    def test_parse_all_cameras(self):
        self.assertEqual(parse_cam_ids('all', 6), [0, 1, 2, 3, 4, 5])

    def test_parse_camera_list_deduplicates_and_sorts(self):
        self.assertEqual(parse_cam_ids('4, 1,4,2', 6), [1, 2, 4])

    def test_parse_camera_list_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            parse_cam_ids('front', 6)
        with self.assertRaises(ValueError):
            parse_cam_ids('9', 6)

    def test_patch_mask_selects_requested_area(self):
        grads = torch.zeros((1, 1, 3, 4, 4))
        grads[:, :, :, 2:, 2:] = 10.0
        roi = torch.ones((1, 1, 1, 4, 4))
        mask, chosen = build_single_patch_mask(
            grads, roi, patch_size=2, min_frac=1.0, min_size=1)
        self.assertEqual(float(mask.sum()), 4.0)
        self.assertEqual(chosen[0][0][0], 2)
        self.assertEqual(chosen[0][0][1:3], (2, 2))


class _CheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.with_cp = True
        self.use_checkpoint = True


class _HeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pts_bbox_head = nn.Module()
        self.pts_bbox_head.task_heads = nn.ModuleList([nn.Module()])
        self.pts_bbox_head.task_heads[0].heatmap = nn.Sequential(
            nn.Identity(), nn.Conv2d(1, 1, 1, bias=False))


class HookTests(unittest.TestCase):
    def test_disable_checkpointing(self):
        model = nn.Sequential(_CheckpointModule())
        disable_checkpointing(model)
        self.assertFalse(model[0].with_cp)
        self.assertFalse(model[0].use_checkpoint)

    def test_heatmap_hook_captures_output(self):
        model = _HeatmapModel()
        hooks, holder = register_all_task_heatmap_logits_hooks(model)
        try:
            model.pts_bbox_head.task_heads[0].heatmap(torch.ones(1, 1, 2, 2))
            self.assertIn(0, holder)
            self.assertEqual(tuple(holder[0].shape), (1, 1, 2, 2))
        finally:
            for hook in hooks:
                hook.remove()

    def test_depth_bin_discovery(self):
        model = nn.Module()
        model.img_view_transformer = nn.Module()
        model.img_view_transformer.D = 59
        self.assertEqual(_get_D_from_model(model), 59)


if __name__ == '__main__':
    unittest.main()
