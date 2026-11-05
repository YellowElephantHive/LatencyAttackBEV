"""Dense geometry objectives."""

import torch
import torch.nn.functional as F

def _parse_two_ints(s: str, default=(0, 1)):
    try:
        parts = [p.strip() for p in str(s).split(',') if p.strip()]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return default

def _parse_three_ints(s: str, default=(0, 1, 2)):
    try:
        parts = [p.strip() for p in str(s).split(',') if p.strip()]
        if len(parts) >= 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        pass
    return default

def geom_area_loss_ascent(dim_map: torch.Tensor, gate: torch.Tensor, dim_idx: str, eps: float):
    """Ascent objective for shrinking BEV area: maximize (-mean((w*l)^2))."""
    wi, li = _parse_two_ints(dim_idx, default=(0, 1))
    # ensure positive dims
    w = F.softplus(dim_map[:, wi:wi+1, :, :])
    l = F.softplus(dim_map[:, li:li+1, :, :])
    area2 = (w * l).pow(2)
    # gate -> (B,1,H,W)
    if gate.shape[1] != 1:
        gate = gate.sum(dim=1, keepdim=True)
    num = (area2 * gate).sum()
    den = gate.sum().clamp_min(float(eps))
    mean_area2 = num / den
    return -mean_area2

def geom_volume_loss_ascent(dim_map: torch.Tensor, gate: torch.Tensor, dim_idx: str, eps: float):
    """Ascent objective for shrinking 3D volume: maximize (-mean((w*l*h)^2)).

    Note: In BEVDet/CenterHead, `dim` typically has 3 channels corresponding to box size (w,l,h).
    """
    wi, li, hi = _parse_three_ints(dim_idx, default=(0, 1, 2))
    # ensure positive dims
    w = F.softplus(dim_map[:, wi:wi+1, :, :])
    l = F.softplus(dim_map[:, li:li+1, :, :])
    h = F.softplus(dim_map[:, hi:hi+1, :, :])
    vol2 = (w * l * h).pow(2)
    # gate -> (B,1,H,W)
    if gate.shape[1] != 1:
        gate = gate.sum(dim=1, keepdim=True)
    num = (vol2 * gate).sum()
    den = gate.sum().clamp_min(float(eps))
    mean_vol2 = num / den
    return -mean_vol2

def geom_reg_dispersion_ascent(reg_map: torch.Tensor, gate: torch.Tensor, eps: float):
    """Ascent objective to diversify reg offsets at selected locations (increase variance)."""
    # reg_map: (B,2,H,W)
    if gate.shape[1] != 1:
        gate1 = gate.sum(dim=1, keepdim=True)
    else:
        gate1 = gate
    wsum = gate1.sum().clamp_min(float(eps))

    dx = reg_map[:, 0:1, :, :]
    dy = reg_map[:, 1:2, :, :]

    mux = (dx * gate1).sum() / wsum
    muy = (dy * gate1).sum() / wsum

    var = (((dx - mux) ** 2 + (dy - muy) ** 2) * gate1).sum() / wsum
    return var

def geom_rot_dispersion_ascent(rot_map: torch.Tensor, gate: torch.Tensor, eps: float):
    """Ascent objective to diversify yaw proxy using rot head.

    We treat rot_map's 2 channels as a 2D direction vector (e.g., sin/cos or cos/sin).
    Dispersion proxy: 1 - ||weighted_mean_unit_vector||.
    """
    if gate.shape[1] != 1:
        gate1 = gate.sum(dim=1, keepdim=True)
    else:
        gate1 = gate
    wsum = gate1.sum().clamp_min(float(eps))

    v = rot_map[:, 0:2, :, :]
    v_norm = torch.sqrt((v * v).sum(dim=1, keepdim=True)).clamp_min(float(eps))
    v_unit = v / v_norm

    mx = (v_unit[:, 0:1, :, :] * gate1).sum() / wsum
    my = (v_unit[:, 1:2, :, :] * gate1).sum() / wsum
    r = torch.sqrt(mx * mx + my * my)
    disp = 1.0 - r
    return disp
