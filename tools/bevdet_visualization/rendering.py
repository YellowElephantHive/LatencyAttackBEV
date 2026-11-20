"""Pure rendering helpers for portfolio-ready BEVDet video frames."""

import math

import cv2
import numpy as np


CAM_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
BG = (20, 24, 32)
TEXT = (235, 239, 245)
MUTED = (158, 168, 184)


def _to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_norm_to_bgr_uint8(image):
    """Convert a normalized CHW camera tensor to an OpenCV BGR image."""
    array = _to_numpy(image).astype(np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError('expected a normalized CHW image with three channels')
    array = array.transpose(1, 2, 0)
    array = np.clip(array * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    rgb = np.rint(array * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _label_image(image, label):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (12, 16, 24), -1)
    cv2.putText(result, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, TEXT, 1, cv2.LINE_AA)
    return result


def make_camera_grid(images, labels=CAM_ORDER):
    """Return a 3x2 camera mosaic from NCHW or CHW image tensors."""
    array = images
    if hasattr(array, 'dim') and array.dim() == 5:
        array = array[0]
    elif not hasattr(array, 'dim'):
        array = _to_numpy(array)
        if array.ndim == 5:
            array = array[0]
    if len(array) < 6:
        raise ValueError('six synchronized camera images are required')
    rendered = [
        _label_image(tensor_norm_to_bgr_uint8(array[i]), labels[i])
        for i in range(6)
    ]
    return np.concatenate([
        np.concatenate(rendered[:3], axis=1),
        np.concatenate(rendered[3:6], axis=1),
    ], axis=0)


def make_difference_grid(clean_images, attack_images, amplification=8.0):
    """Render an amplified absolute RGB difference for six cameras."""
    clean = clean_images
    attack = attack_images
    if hasattr(clean, 'dim') and clean.dim() == 5:
        clean = clean[0]
    elif not hasattr(clean, 'dim') and _to_numpy(clean).ndim == 5:
        clean = _to_numpy(clean)[0]
    if hasattr(attack, 'dim') and attack.dim() == 5:
        attack = attack[0]
    elif not hasattr(attack, 'dim') and _to_numpy(attack).ndim == 5:
        attack = _to_numpy(attack)[0]
    rendered = []
    for index in range(6):
        clean_bgr = tensor_norm_to_bgr_uint8(clean[index]).astype(np.float32)
        attack_bgr = tensor_norm_to_bgr_uint8(attack[index]).astype(np.float32)
        diff = np.clip(np.abs(attack_bgr - clean_bgr) * amplification, 0, 255).astype(np.uint8)
        rendered.append(_label_image(diff, CAM_ORDER[index]))
    return np.concatenate([
        np.concatenate(rendered[:3], axis=1),
        np.concatenate(rendered[3:6], axis=1),
    ], axis=0)


def box_corners_bev(box):
    """Return the four XY corners for a decoded CenterPoint box."""
    x, y, _, width, length, _, yaw = [float(v) for v in box[:7]]
    local = np.asarray([
        [-width / 2.0, -length / 2.0],
        [width / 2.0, -length / 2.0],
        [width / 2.0, length / 2.0],
        [-width / 2.0, length / 2.0],
    ], dtype=np.float32)
    rotation = np.asarray([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)],
    ], dtype=np.float32)
    return local @ rotation.T + np.asarray([x, y], dtype=np.float32)


def _world_to_pixel(points, size, extent):
    minimum, maximum = extent
    result = np.empty_like(points, dtype=np.float32)
    result[:, 0] = (points[:, 0] - minimum) / (maximum - minimum) * (size - 1)
    result[:, 1] = (maximum - points[:, 1]) / (maximum - minimum) * (size - 1)
    return np.rint(result).astype(np.int32)


def render_bev_candidates(boxes, scores, selected=None, ground_truth=None,
                          size=512, extent=(-51.2, 51.2), draw_limit=500,
                          candidate_color=(55, 105, 245)):
    """Render decoded pre-NMS candidates and selected boxes in bird's-eye view."""
    boxes = _to_numpy(boxes).reshape(-1, _to_numpy(boxes).shape[-1]) if len(boxes) else np.empty((0, 9))
    scores = _to_numpy(scores).reshape(-1) if len(scores) else np.empty((0,))
    selected_set = set(int(v) for v in (_to_numpy(selected).reshape(-1) if selected is not None else []))
    canvas = np.full((size, size, 3), BG, dtype=np.uint8)
    for meter in range(-50, 51, 10):
        p0 = _world_to_pixel(np.asarray([[meter, extent[0]], [meter, extent[1]]], dtype=np.float32), size, extent)
        p1 = _world_to_pixel(np.asarray([[extent[0], meter], [extent[1], meter]], dtype=np.float32), size, extent)
        cv2.line(canvas, tuple(p0[0]), tuple(p0[1]), (42, 48, 60), 1)
        cv2.line(canvas, tuple(p1[0]), tuple(p1[1]), (42, 48, 60), 1)

    if ground_truth is not None:
        for box in _to_numpy(ground_truth):
            if len(box) < 7 or not np.isfinite(box[:7]).all():
                continue
            corners = _world_to_pixel(box_corners_bev(box), size, extent)
            cv2.polylines(canvas, [corners], True, (235, 235, 235), 2, cv2.LINE_AA)

    if len(scores):
        order = np.argsort(-scores)[:max(0, int(draw_limit))]
        overlay = canvas.copy()
        for index in order[::-1]:
            box = boxes[index]
            if len(box) < 7 or not np.isfinite(box[:7]).all():
                continue
            if box[3] <= 0 or box[4] <= 0 or box[3] > 30 or box[4] > 30:
                continue
            corners = _world_to_pixel(box_corners_bev(box), size, extent)
            color = (72, 214, 126) if int(index) in selected_set else candidate_color
            thickness = 2 if int(index) in selected_set else 1
            cv2.polylines(overlay, [corners], True, color, thickness, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0.0)

    ego = _world_to_pixel(np.asarray([[0.0, 0.0]], dtype=np.float32), size, extent)[0]
    cv2.circle(canvas, tuple(ego), 6, (255, 210, 80), -1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, tuple(ego), (int(ego[0]), int(ego[1] - 35)),
                    (255, 210, 80), 2, cv2.LINE_AA, tipLength=0.3)
    return canvas


def _fit(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                 max(1, int(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    result = np.full((height, width, 3), BG, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    result[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return result


def _title(canvas, text, x, y, color=TEXT, scale=0.72):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def compose_comparison_frame(clean_grid, attack_grid, difference_grid,
                             clean_bev, attack_bev, clean_count, attack_count,
                             sample_token, frame_index, total_frames):
    """Compose one 1920x1080 headless video frame."""
    canvas = np.full((1080, 1920, 3), BG, dtype=np.uint8)
    _title(canvas, 'BEVDet Candidate Flood Under Gate-Sum Attack', 28, 38, scale=0.88)
    _title(canvas, f'Frame {frame_index + 1}/{total_frames}  sample {sample_token[:8]}',
           1280, 38, color=MUTED, scale=0.58)

    canvas[70:500, 20:940] = _fit(clean_grid, 920, 430)
    canvas[70:500, 980:1900] = _fit(attack_grid, 920, 430)
    _title(canvas, 'Clean camera inputs', 28, 66)
    _title(canvas, 'Attacked camera inputs', 988, 66, color=(100, 145, 255))

    canvas[580:1040, 20:940] = _fit(difference_grid, 920, 460)
    canvas[580:1040, 980:1430] = _fit(clean_bev, 450, 460)
    canvas[580:1040, 1450:1900] = _fit(attack_bev, 450, 460)
    _title(canvas, 'Amplified perturbation (8x)', 28, 570)
    _title(canvas, f'Clean pre-NMS: {clean_count:,}', 988, 570,
           color=(245, 170, 75))
    _title(canvas, f'Attack pre-NMS: {attack_count:,}', 1458, 570,
           color=(80, 120, 255))
    return canvas
