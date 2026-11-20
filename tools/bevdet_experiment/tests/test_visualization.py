import unittest

import numpy as np

from bevdet_visualization.rendering import (
    box_corners_bev,
    compose_comparison_frame,
    make_camera_grid,
    make_difference_grid,
    render_bev_candidates,
)


class VisualizationTests(unittest.TestCase):
    def test_camera_and_difference_grid_shapes(self):
        images = np.zeros((1, 6, 3, 32, 64), dtype=np.float32)
        attacked = images.copy()
        attacked[:, 1] += 0.01
        self.assertEqual(make_camera_grid(images).shape, (64, 192, 3))
        self.assertEqual(
            make_difference_grid(images, attacked).shape,
            (64, 192, 3),
        )

    def test_box_corners_and_bev_render(self):
        boxes = np.asarray([
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],
            [10.0, 5.0, 0.0, 2.0, 2.0, 1.5, 0.5, 0.0, 0.0],
        ], dtype=np.float32)
        corners = box_corners_bev(boxes[0])
        self.assertEqual(corners.shape, (4, 2))
        image = render_bev_candidates(
            boxes, np.asarray([0.9, 0.5]), selected=np.asarray([0]),
            ground_truth=boxes[:1], size=256,
        )
        self.assertEqual(image.shape, (256, 256, 3))
        self.assertGreater(int(image.sum()), 0)

    def test_composed_frame_is_full_hd(self):
        camera = np.zeros((64, 192, 3), dtype=np.uint8)
        bev = np.zeros((128, 128, 3), dtype=np.uint8)
        result = compose_comparison_frame(
            camera, camera, camera, bev, bev,
            800, 2100, 'abc12345', 0, 10,
        )
        self.assertEqual(result.shape, (1080, 1920, 3))


if __name__ == '__main__':
    unittest.main()
