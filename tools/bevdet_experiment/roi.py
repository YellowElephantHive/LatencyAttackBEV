"""Camera selection and ROI constraint helpers."""

import os

import cv2
import numpy as np
import torch

CAM_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

def _basename_no_ext(path_str: str) -> str:
    base = os.path.basename(path_str)
    stem, _ = os.path.splitext(base)
    return stem

def load_roi_mask_tensor(sample_info, cam_order, data_root, roi_mask_dir, H, W, device,
                         thr=127, missing='all'):
    """Load per-camera ROI masks into a tensor (B=1,N,1,H,W) with values in {0,1}."""
    out = torch.ones((1, len(cam_order), 1, H, W), device=device, dtype=torch.float32)

    if sample_info is None or 'cams' not in sample_info:
        return out

    cams = sample_info['cams']
    for n, cam_name in enumerate(cam_order):
        cam_meta = cams.get(cam_name, None)
        if cam_meta is None:
            continue
        data_path = cam_meta.get('data_path', None)
        if not data_path:
            continue

        stem = _basename_no_ext(data_path)
        mask_path = os.path.join(data_root, roi_mask_dir, cam_name, stem + '.png')

        if not os.path.isfile(mask_path):
            if missing == 'error':
                raise FileNotFoundError(f"ROI mask missing: {mask_path}")
            elif missing == 'zero':
                out[:, n, :, :, :] = 0.0
            continue

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            if missing == 'error':
                raise RuntimeError(f"Failed to read ROI mask: {mask_path}")
            elif missing == 'zero':
                out[:, n, :, :, :] = 0.0
            continue

        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        m01 = (m.astype(np.uint8) >= int(thr)).astype(np.float32)
        out[:, n, 0, :, :] = torch.from_numpy(m01).to(device=device, dtype=torch.float32)

    return out

def build_single_patch_mask(grads, roi_mask, patch_size=64, min_frac=0.6, min_size=8,
                            fallback='shrink'):
    """Select a single patch with highest grad energy inside ROI; return patch_mask (B,N,1,H,W)."""
    assert grads is not None and roi_mask is not None
    B, N, C, H, W = grads.shape
    device = grads.device

    patch_mask = torch.zeros((B, N, 1, H, W), device=device, dtype=torch.float32)
    chosen = [[None for _ in range(N)] for _ in range(B)]

    g = grads.abs().mean(dim=2, keepdim=True)  # (B,N,1,H,W)
    roi = roi_mask.float()

    for b in range(B):
        for n in range(N):
            roi_bn = roi[b:b+1, n:n+1]
            if float(roi_bn.sum().item()) < 1.0:
                chosen[b][n] = ('empty_roi', 0, 0, 0)
                continue

            s = int(patch_size)
            found = False

            while True:
                s = min(s, H, W)
                if s < int(min_size):
                    break

                g_bn = g[b:b+1, n:n+1, 0]       # (1,1,H,W)
                roi_bn_2d = roi_bn[:, :, 0]     # (1,1,H,W)

                kernel = torch.ones((1, 1, s, s), device=device, dtype=torch.float32)

                score_map = torch.nn.functional.conv2d(g_bn * roi_bn_2d, kernel, stride=1, padding=0)
                area_map  = torch.nn.functional.conv2d(roi_bn_2d, kernel, stride=1, padding=0)

                need = float(min_frac) * float(s * s)
                valid = (area_map >= need)

                if valid.any():
                    masked_score = score_map.clone()
                    masked_score[~valid] = -1e9

                    flat_idx = int(masked_score.view(-1).argmax().item())
                    out_h, out_w = masked_score.shape[-2], masked_score.shape[-1]
                    yy = flat_idx // out_w
                    xx = flat_idx % out_w

                    patch_mask[b, n, 0, yy:yy+s, xx:xx+s] = 1.0
                    chosen[b][n] = (s, int(yy), int(xx), float(area_map[0,0,yy,xx].item()))
                    found = True
                    break

                if fallback == 'shrink':
                    s = s // 2
                    continue
                elif fallback == 'skip':
                    chosen[b][n] = ('skip_no_valid', s, 0, 0)
                    break
                else:
                    raise RuntimeError(f"No valid patch found (b={b}, n={n}) at size={s}. ROI may be too small.")

            if not found and chosen[b][n] is None:
                chosen[b][n] = ('no_valid', s, 0, 0)

    return patch_mask, chosen

def parse_cam_ids(attack_cams: str, N_cam: int):
    if attack_cams.lower() == 'all':
        return list(range(N_cam))
    cam_ids = []
    for token in attack_cams.split(','):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Non-numeric attack_cams token: {token}")
        cid = int(token)
        if 0 <= cid < N_cam:
            cam_ids.append(cid)
    cam_ids = sorted(set(cam_ids))
    if not cam_ids:
        raise ValueError(f"No valid camera indices in --attack-cams='{attack_cams}' for N_cam={N_cam}")
    return cam_ids
