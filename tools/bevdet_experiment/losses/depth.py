"""Depth objectives used by the adversarial attack."""

import torch

def _depth_prob_from_depthnet_out(x_depthnet, D, eps=1e-12):
    if x_depthnet is None or (not torch.is_tensor(x_depthnet)):
        return None
    if x_depthnet.dim() != 4:
        return None
    if x_depthnet.shape[1] < D:
        return None
    x_depth = x_depthnet[:, :D, :, :]
    prob = torch.softmax(x_depth, dim=1).clamp_min(eps)
    return prob

def depth_expected_bin_and_entropy(prob):
    D = prob.shape[1]
    k = torch.arange(D, device=prob.device, dtype=prob.dtype).view(1, D, 1, 1)
    Ek = (prob * k).sum(dim=1)
    ent = -(prob * prob.log()).sum(dim=1)
    return Ek, ent

def depth_loss_entropy_from_x(x_depthnet, D, eps=1e-12):
    prob = _depth_prob_from_depthnet_out(x_depthnet, D, eps=eps)
    if prob is None:
        return None
    ent = -(prob * prob.log()).sum(dim=1)
    return ent.mean()

def depth_loss_logp_fixed_from_x(x_depthnet, D, kstar=-1, eps=1e-12):
    prob = _depth_prob_from_depthnet_out(x_depthnet, D, eps=eps)
    if prob is None:
        return None
    k = int(kstar)
    if k < 0:
        k = D - 1
    k = max(0, min(D - 1, k))
    return prob[:, k, :, :].log().mean()

def depth_loss_logp_shift_from_x(x_depthnet, D, kdelta=0, eps=1e-12):
    prob = _depth_prob_from_depthnet_out(x_depthnet, D, eps=eps)
    if prob is None:
        return None
    k_hat = prob.argmax(dim=1)
    k_tgt = (k_hat + int(kdelta)).clamp(0, D - 1)
    p_tgt = torch.gather(prob, dim=1, index=k_tgt.unsqueeze(1)).squeeze(1)
    return p_tgt.log().mean()

def depth_loss_ek_shift_from_prob(prob, Ek0, ent0, delta, constrained=False, beta=1.0):
    Ek, ent = depth_expected_bin_and_entropy(prob)
    D = prob.shape[1]
    Ek_tgt = (Ek0 + float(delta)).clamp(0.0, float(D - 1))
    shift_term = - (Ek - Ek_tgt).pow(2).mean()
    if not constrained:
        return shift_term
    shape_term = - float(beta) * (ent - ent0).pow(2).mean()
    return shift_term + shape_term

def depth_loss_multi_peak_from_x(x_depthnet, D, K=2.0, steep=10.0, margin=0.0, sharp=0.0):
    if x_depthnet is None or (not torch.is_tensor(x_depthnet)):
        return None
    if x_depthnet.dim() != 4 or x_depthnet.shape[1] < D:
        return None

    z = x_depthnet[:, :D, :, :]
    s = float(steep)
    m = float(margin)

    z_left = torch.cat([z[:, :1, :, :], z[:, :-1, :, :]], dim=1)
    z_right = torch.cat([z[:, 1:, :, :], z[:, -1:, :, :]], dim=1)

    gL = torch.sigmoid(s * (z - z_left - m))
    gR = torch.sigmoid(s * (z - z_right - m))

    gL = torch.cat([torch.ones_like(gL[:, :1, :, :]), gL[:, 1:, :, :]], dim=1)
    gR = torch.cat([gR[:, :-1, :, :], torch.ones_like(gR[:, -1:, :, :])], dim=1)

    m_pk = gL * gR
    Khat = m_pk.sum(dim=1)

    loss = - (Khat - float(K)).pow(2).mean()
    if sharp and float(sharp) != 0.0:
        loss = loss + float(sharp) * (m_pk * z).mean()
    return loss
