"""Heatmap surrogate objectives."""

import torch
import torch.nn.functional as F

def hm_loss_gate_sum(hm_logits_by_task, tau: float, steep: float, device):
    """Original surrogate: encourage many pixels above tau (area-like)."""
    if not hm_logits_by_task:
        return torch.tensor(0.0, device=device)
    acc = 0.0
    cnt = 0
    for _, z in sorted(hm_logits_by_task.items()):
        prob = torch.sigmoid(z)
        gate = torch.sigmoid(float(steep) * (prob - float(tau)))
        acc = acc + gate.sum()
        cnt += 1
    return acc / max(1, cnt)

def hm_loss_overload_hardish(hm_logits_by_task, tau: float, steep: float, device):
    """Overload-style hard-ish surrogate: maximize confidence mass above a threshold.

    Implements: sum_u 1[p(u)>tau] * p(u), with a sigmoid gate as a smooth indicator.
    gate = sigmoid(steep * (p - tau)); objective = sum(gate * p)
    """
    if not hm_logits_by_task:
        return torch.tensor(0.0, device=device)
    acc = 0.0
    cnt = 0
    for _, z in sorted(hm_logits_by_task.items()):
        prob = torch.sigmoid(z)
        gate = torch.sigmoid(float(steep) * (prob - float(tau)))
        acc = acc + (gate * prob).sum()
        cnt += 1
    return acc / max(1, cnt)

def hm_loss_peak_count(hm_logits_by_task, tau: float, gate_steep: float,
                       peak_kernel: int, peak_steep: float, peak_eps: float,
                       device):
    """Peak-count surrogate: encourage MORE local maxima above tau.

    Soft peakness (maxpool-based):
      prob = sigmoid(z)
      pooled = maxpool(prob)
      peakness = sigmoid(peak_steep * (prob - pooled + peak_eps))
      gate = sigmoid(gate_steep * (prob - tau))
      maximize sum(peakness * gate)
    """
    if not hm_logits_by_task:
        return torch.tensor(0.0, device=device)

    k = int(peak_kernel)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k = k + 1
    pad = k // 2

    acc = 0.0
    cnt = 0
    for _, z in sorted(hm_logits_by_task.items()):
        prob = torch.sigmoid(z)
        pooled = F.max_pool2d(prob, kernel_size=k, stride=1, padding=pad)
        peakness = torch.sigmoid(float(peak_steep) * (prob - pooled + float(peak_eps)))
        gate = torch.sigmoid(float(gate_steep) * (prob - float(tau)))
        acc = acc + (peakness * gate).sum()
        cnt += 1
    return acc / max(1, cnt)

def hm_loss_topk_sum(hm_logits_by_task, topk: int, device):
    """TopK surrogate: maximize sum of topK probs per class (spatial)."""
    if not hm_logits_by_task:
        return torch.tensor(0.0, device=device)

    K = int(topk)
    acc = 0.0
    cnt = 0
    for _, z in sorted(hm_logits_by_task.items()):
        prob = torch.sigmoid(z)  # (B,C,H,W)
        B, C, H, W = prob.shape
        flat = prob.view(B, C, -1)
        K_eff = min(K, flat.shape[-1])
        vals, _ = torch.topk(flat, k=K_eff, dim=-1, largest=True, sorted=False)
        acc = acc + vals.sum()
        cnt += 1
    return acc / max(1, cnt)
