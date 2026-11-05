"""Model hook registration helpers."""

import torch
import torch.nn as nn

def disable_checkpointing(model):
    """Disable gradient checkpointing to avoid torch.utils.checkpoint tuple errors during PGD backward."""
    for m in model.modules():
        if hasattr(m, 'with_cp'):
            try:
                m.with_cp = False
            except Exception:
                pass
        if hasattr(m, 'use_checkpoint'):
            try:
                m.use_checkpoint = False
            except Exception:
                pass

def register_all_task_heatmap_logits_hooks(model, max_tasks=32):
    """Hook all: pts_bbox_head.task_heads.{t}.heatmap.1 (final heatmap logits)."""
    holders = {}
    hooks = []

    def make_hook(tid):
        def _hook(m, inp, out):
            holders[tid] = out
            if torch.is_tensor(out) and out.requires_grad:
                out.retain_grad()
        return _hook

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if ".task_heads." in name and ".heatmap.1" in name:
            try:
                tid = int(name.split(".task_heads.")[1].split(".")[0])
            except Exception:
                continue
            if tid < max_tasks:
                hooks.append(module.register_forward_hook(make_hook(tid)))

    return hooks, holders

def register_all_task_head_dict_hooks(model, max_tasks=32):
    """Hook `pts_bbox_head.task_heads.{tid}` outputs (dict with keys like reg/dim/rot/heatmap).

    This provides differentiable access to regression maps (dim/reg/rot) for geometry losses.
    """
    holders = {}
    hooks = []

    def make_hook(tid):
        def _hook(m, inp, out):
            holders[tid] = out
            if isinstance(out, dict):
                for _, v in out.items():
                    if torch.is_tensor(v) and v.requires_grad:
                        v.retain_grad()
        return _hook

    for name, module in model.named_modules():
        if '.task_heads.' not in name:
            continue
        # We only want to hook the task head module itself, e.g. "pts_bbox_head.task_heads.0"
        try:
            tid = int(name.split('.task_heads.')[1].split('.')[0])
        except Exception:
            continue
        if tid >= max_tasks:
            continue
        if name.endswith(f'.task_heads.{tid}') or name.endswith(f'task_heads.{tid}'):
            hooks.append(module.register_forward_hook(make_hook(tid)))

    return hooks, holders

def _get_D_from_model(model):
    ivt = getattr(model, 'img_view_transformer', None)
    if ivt is None:
        return None
    if hasattr(ivt, 'D'):
        return int(ivt.D)
    if hasattr(ivt, 'frustum'):
        return int(ivt.frustum.shape[0])
    return None

def register_depthnet_hook(model):
    """Hook img_view_transformer.depthnet output x: (B*N, D+numC_Trans, H, W)."""
    holder = {'x': None, 'name': None}

    def _hook(m, inp, out):
        holder['x'] = out
        if torch.is_tensor(out) and out.requires_grad:
            out.retain_grad()

    for name, module in model.named_modules():
        if name.endswith('img_view_transformer.depthnet'):
            h = module.register_forward_hook(_hook)
            holder['name'] = name
            return h, holder, name

    for name, module in model.named_modules():
        n = name.lower()
        if ('img_view_transformer' in n) and ('depthnet' in n):
            try:
                h = module.register_forward_hook(_hook)
                holder['name'] = name
                return h, holder, name
            except Exception:
                pass

    return None, holder, None
