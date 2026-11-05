"""Proposal decoding and proposal-level objectives."""

import torch
import torch.nn.functional as F

def gather_topk_proposals_from_task(out_d, hm_z, topk=200, score_mode='maxcls'):
    """
    out_d: dict with keys reg, height, dim, rot, heatmap(optional)
    hm_z:  heatmap logits tensor, shape (B,C,H,W)
    Return:
        props: dict of gathered tensors at top-K locations
    """
    prob = torch.sigmoid(hm_z)   # (B,C,H,W)
    B, C, H, W = prob.shape

    if score_mode == 'maxcls':
        score_map, cls_map = prob.max(dim=1)   # (B,H,W)
    else:
        score_map = prob.sum(dim=1)
        cls_map = None

    flat_score = score_map.view(B, -1)
    K = min(int(topk), flat_score.shape[1])
    topk_vals, topk_idx = torch.topk(flat_score, k=K, dim=1, largest=True, sorted=False)

    ys = topk_idx // W
    xs = topk_idx % W

    def _gather_map(m):
        # m: (B,Cm,H,W)
        Bm, Cm, Hm, Wm = m.shape
        flat = m.view(Bm, Cm, -1)
        idx = topk_idx.unsqueeze(1).expand(Bm, Cm, K)
        val = torch.gather(flat, dim=2, index=idx)   # (B,Cm,K)
        return val

    props = {
        'score': topk_vals,          # (B,K)
        'x_idx': xs.float(),         # (B,K)
        'y_idx': ys.float(),         # (B,K)
    }

    for key in ['reg', 'height', 'dim', 'rot']:
        if key in out_d and torch.is_tensor(out_d[key]):
            props[key] = _gather_map(out_d[key])   # (B,Cm,K)

    return props

def decode_topk_props_simple(props, pc_range=None, voxel_size=None, out_size_factor=None,
                             norm_bbox=True, eps=1e-6):
    """
    Decode gathered top-K proposal attributes into approximate box parameters.
    Return:
        cx, cy, z, l, w, h, yaw, score
        all shape: (B,K)
    """
    score = props['score']                      # (B,K)
    x_idx = props['x_idx']
    y_idx = props['y_idx']

    reg = props.get('reg', None)               # (B,2,K)
    height = props.get('height', None)         # (B,1,K)
    dim = props.get('dim', None)               # (B,3,K)
    rot = props.get('rot', None)               # (B,2,K)

    if reg is not None and reg.shape[1] >= 2:
        dx = reg[:, 0, :]
        dy = reg[:, 1, :]
    else:
        dx = torch.zeros_like(x_idx)
        dy = torch.zeros_like(y_idx)

    cx = x_idx + dx
    cy = y_idx + dy

    if voxel_size is not None and out_size_factor is not None and pc_range is not None:
        vx, vy = float(voxel_size[0]), float(voxel_size[1])
        ofs = float(out_size_factor)
        cx = cx * ofs * vx + float(pc_range[0])
        cy = cy * ofs * vy + float(pc_range[1])

    if height is not None:
        z = height[:, 0, :]
    else:
        z = torch.zeros_like(cx)

    if dim is not None and dim.shape[1] >= 3:
        d0 = dim[:, 0, :]
        d1 = dim[:, 1, :]
        d2 = dim[:, 2, :]
        if norm_bbox:
            d0 = torch.exp(d0)
            d1 = torch.exp(d1)
            d2 = torch.exp(d2)
        else:
            d0 = F.softplus(d0) + eps
            d1 = F.softplus(d1) + eps
            d2 = F.softplus(d2) + eps

        # 你可依 repo 慣例改順序
        w = d0
        l = d1
        h = d2
    else:
        w = torch.ones_like(cx)
        l = torch.ones_like(cx)
        h = torch.ones_like(cx)

    if rot is not None and rot.shape[1] >= 2:
        yaw = torch.atan2(rot[:, 0, :], rot[:, 1, :])
    else:
        yaw = torch.zeros_like(cx)

    return {
        'score': score,
        'cx': cx,
        'cy': cy,
        'z': z,
        'l': l,
        'w': w,
        'h': h,
        'yaw': yaw,
    }

def proposal_shape_range_loss_ascent(boxes, args):
    l = boxes['l']
    w = boxes['w']
    h = boxes['h']

    loss = (
        F.relu(l - float(args.shape_lmax)) +
        F.relu(float(args.shape_lmin) - l) +
        F.relu(w - float(args.shape_wmax)) +
        F.relu(float(args.shape_wmin) - w) +
        F.relu(h - float(args.shape_hmax)) +
        F.relu(float(args.shape_hmin) - h)
    ).mean()

    return -loss

def proposal_vertical_loss_ascent(boxes, args):
    z = boxes['z']
    loss = (
        F.relu(float(args.vert_zmin) - z) +
        F.relu(z - float(args.vert_zmax))
    ).mean()
    return -loss

def proposal_diversity_loss_ascent(boxes, args):
    cx = boxes['cx']   # (B,K)
    cy = boxes['cy']
    B, K = cx.shape

    K_use = min(K, int(args.div_sample_k))
    cx = cx[:, :K_use]
    cy = cy[:, :K_use]

    pts = torch.stack([cx, cy], dim=-1)    # (B,K,2)
    diff = pts.unsqueeze(2) - pts.unsqueeze(1)   # (B,K,K,2)
    dist2 = (diff ** 2).sum(dim=-1)               # (B,K,K)

    eye = torch.eye(K_use, device=dist2.device).unsqueeze(0)
    dist = torch.sqrt(dist2 + 1e-6)

    penalty = F.relu(float(args.div_min_dist) - dist)
    penalty = penalty * (1.0 - eye)

    loss = penalty.sum() / ((1.0 - eye).sum().clamp_min(1.0))
    return -loss
