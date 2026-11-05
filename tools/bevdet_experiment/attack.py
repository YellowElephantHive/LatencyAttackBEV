"""PGD attack orchestration for a single BEVDet batch."""

import numpy as np
import torch

from .counting import find_center_head
from .losses.depth import (
    _depth_prob_from_depthnet_out,
    depth_expected_bin_and_entropy,
    depth_loss_ek_shift_from_prob,
    depth_loss_entropy_from_x,
    depth_loss_logp_fixed_from_x,
    depth_loss_logp_shift_from_x,
    depth_loss_multi_peak_from_x,
)
from .losses.geometry import (
    geom_area_loss_ascent,
    geom_reg_dispersion_ascent,
    geom_rot_dispersion_ascent,
    geom_volume_loss_ascent,
)
from .losses.heatmap import (
    hm_loss_gate_sum,
    hm_loss_overload_hardish,
    hm_loss_peak_count,
    hm_loss_topk_sum,
)
from .losses.proposal import (
    decode_topk_props_simple,
    gather_topk_proposals_from_task,
    proposal_diversity_loss_ascent,
    proposal_shape_range_loss_ascent,
    proposal_vertical_loss_ascent,
)
from .roi import (
    CAM_ORDER,
    build_single_patch_mask,
    load_roi_mask_tensor,
    parse_cam_ids,
)

def attack_one_batch_phase1_hmonly(model, batch, args, device, hm_task_holder,
                                  task_head_holder=None,
                                  sample_info=None, data_root=None,
                                  D=None, depth_holder=None):
    """Phase-1: HM-only + attack-cams + ROI mask + ROI patch (no image saving)."""
    img_inputs = batch.get('img_inputs', None)
    if hasattr(img_inputs, 'data'):
        img_inputs = img_inputs.data

    inner = img_inputs[0]
    imgs = inner[0]
    assert torch.is_tensor(imgs) and imgs.dim() == 5
    imgs = imgs.to(device)

    imgs_orig = imgs.detach().clone()
    imgs.requires_grad_(True)

    # write-back handle
    inner = list(inner); inner[0] = imgs
    img_inputs = list(img_inputs); img_inputs[0] = inner
    batch['img_inputs'] = img_inputs

    B, N_cam, C, H, W = imgs.shape
    cam_ids = parse_cam_ids(args.attack_cams, N_cam)

    roi_mask = None
    if args.roi_mask:
        if data_root is None:
            raise RuntimeError("--roi-mask requires data_root")
        roi_mask = load_roi_mask_tensor(
            sample_info=sample_info,
            cam_order=CAM_ORDER,
            data_root=data_root,
            roi_mask_dir=args.roi_mask_dir,
            H=H, W=W,
            device=device,
            thr=args.roi_mask_thr,
            missing=args.roi_mask_missing,
        )

    imgs_base = imgs_orig
    # Cache clean Ek0/ent0 for ek_shift variants
    Ek0_ref, ent0_ref = None, None
    if (args.lambda_depth > 0) and (args.depth_loss_type in ('ek_shift', 'ek_shift_constrained')):
        with torch.no_grad():
            inner_tmp = list(batch['img_inputs'][0])
            inner_tmp[0] = imgs_orig
            img_inputs_tmp = list(batch['img_inputs'])
            img_inputs_tmp[0] = inner_tmp
            batch_tmp = dict(batch)
            batch_tmp['img_inputs'] = img_inputs_tmp
            _ = model(return_loss=False, rescale=True, **batch_tmp)
            x0 = depth_holder.get('x', None) if depth_holder is not None else None
            prob0 = _depth_prob_from_depthnet_out(x0, D, eps=args.depth_eps) if (x0 is not None and D is not None) else None
            if prob0 is not None:
                Ek0_ref, ent0_ref = depth_expected_bin_and_entropy(prob0)

    # --- grad-stat cosine history across steps (per batch) ---
    cos_hist = {
        'shape': [],
        'vertical': [],
        'div': [],
        'area': [],
        'vol':  [],
        'reg':  [],
        'rot':  [],
        'depth': [],
    }

    for step in range(args.steps):
        # roi-mask-input forward/backward
        if args.roi_mask and args.roi_mask_input and (roi_mask is not None):
            fill_val = 0.0
            imgs_eff = imgs * roi_mask + fill_val * (1.0 - roi_mask)

            inner_eff = list(batch['img_inputs'][0]); inner_eff[0] = imgs_eff
            img_inputs_eff = list(batch['img_inputs']); img_inputs_eff[0] = inner_eff
            batch_eff = dict(batch); batch_eff['img_inputs'] = img_inputs_eff

            _ = model(return_loss=False, rescale=True, **batch_eff)
        else:
            _ = model(return_loss=False, rescale=True, **batch)

        # HM-only loss: all tasks, all classes (variant selectable)
        hm_logits = {tid: t for tid, t in hm_task_holder.items() if torch.is_tensor(t)}
        if args.hm_loss_mode == 'gate_sum':
            L_hm_pos = hm_loss_gate_sum(
                hm_logits_by_task=hm_logits,
                tau=float(args.hm_tau),
                steep=float(args.hm_steep),
                device=device,
            )
        elif args.hm_loss_mode == 'overload_hardish':
            L_hm_pos = hm_loss_overload_hardish(
                hm_logits_by_task=hm_logits,
                tau=float(args.hm_tau),
                steep=float(args.hm_steep),
                device=device,
            )
        elif args.hm_loss_mode == 'peak_count':
            L_hm_pos = hm_loss_peak_count(
                hm_logits_by_task=hm_logits,
                tau=float(args.hm_tau),
                gate_steep=float(args.hm_steep),
                peak_kernel=int(args.hm_peak_kernel),
                peak_steep=float(args.hm_peak_steep),
                peak_eps=float(args.hm_peak_eps),
                device=device,
            )
        elif args.hm_loss_mode == 'topk_sum':
            L_hm_pos = hm_loss_topk_sum(
                hm_logits_by_task=hm_logits,
                topk=int(args.hm_topk),
                device=device,
            )
        else:
            L_hm_pos = torch.tensor(0.0, device=device)

        # Geometry losses (optional)
        L_area = torch.tensor(0.0, device=device)
        L_vol  = torch.tensor(0.0, device=device)
        L_reg  = torch.tensor(0.0, device=device)
        L_rot  = torch.tensor(0.0, device=device)

        if task_head_holder is not None and (
            float(getattr(args, 'lambda_area', 0.0)) > 0.0 or
            float(getattr(args, 'lambda_vol', 0.0)) > 0.0 or
            float(getattr(args, 'lambda_reg', 0.0)) > 0.0 or
            float(getattr(args, 'lambda_rot', 0.0)) > 0.0
        ):
            # Build the same kind of soft gate from heatmap logits
            # (focus geometry objectives on candidate-rich locations)
            tau = float(args.hm_tau)
            steep = float(args.hm_steep)
            eps_g = float(getattr(args, 'geom_eps', 1e-6))

            for tid, out_d in task_head_holder.items():
                if not isinstance(out_d, dict):
                    continue
                hm_z = hm_logits.get(tid, None)
                if hm_z is None:
                    continue
                prob = torch.sigmoid(hm_z)
                gate = torch.sigmoid(steep * (prob - tau))

                if float(getattr(args, 'lambda_area', 0.0)) > 0.0:
                    dim_map = out_d.get('dim', None)
                    if torch.is_tensor(dim_map) and dim_map.dim() == 4 and dim_map.shape[1] >= 2:
                        L_area = L_area + geom_area_loss_ascent(
                            dim_map=dim_map,
                            gate=gate,
                            dim_idx=str(getattr(args, 'area_dim_idx', '0,1')),
                            eps=eps_g,
                        )

                if float(getattr(args, 'lambda_vol', 0.0)) > 0.0:
                    dim_map = out_d.get('dim', None)
                    if torch.is_tensor(dim_map) and dim_map.dim() == 4 and dim_map.shape[1] >= 3:
                        L_vol = L_vol + geom_volume_loss_ascent(
                            dim_map=dim_map,
                            gate=gate,
                            dim_idx=str(getattr(args, 'vol_dim_idx', '0,1,2')),
                            eps=eps_g,
                        )

                if float(getattr(args, 'lambda_reg', 0.0)) > 0.0:
                    reg_map = out_d.get('reg', None)
                    if torch.is_tensor(reg_map) and reg_map.dim() == 4 and reg_map.shape[1] >= 2:
                        L_reg = L_reg + geom_reg_dispersion_ascent(
                            reg_map=reg_map,
                            gate=gate,
                            eps=eps_g,
                        )

                if float(getattr(args, 'lambda_rot', 0.0)) > 0.0:
                    rot_map = out_d.get('rot', None)
                    if torch.is_tensor(rot_map) and rot_map.dim() == 4 and rot_map.shape[1] >= 2:
                        L_rot = L_rot + geom_rot_dispersion_ascent(
                            rot_map=rot_map,
                            gate=gate,
                            eps=eps_g,
                        )

            # average over tasks we accumulated (avoid scaling with number of tasks)
            n_task_eff = max(1, len(task_head_holder))
            L_area = L_area / float(n_task_eff)
            L_vol  = L_vol  / float(n_task_eff)
            L_reg  = L_reg  / float(n_task_eff)
            L_rot  = L_rot  / float(n_task_eff)

        L_depth = torch.tensor(0.0, device=device)
        if (args.lambda_depth > 0) and (depth_holder is not None) and (D is not None):
            x_depthnet = depth_holder.get('x', None)
            if args.depth_loss_type == 'entropy':
                v = depth_loss_entropy_from_x(x_depthnet, D, eps=args.depth_eps)
                if v is not None:
                    L_depth = v
            elif args.depth_loss_type == 'kl_fixed':
                v = depth_loss_logp_fixed_from_x(x_depthnet, D, kstar=args.depth_kstar, eps=args.depth_eps)
                if v is not None:
                    L_depth = v
            elif args.depth_loss_type == 'kl_shift':
                v = depth_loss_logp_shift_from_x(x_depthnet, D, kdelta=args.depth_kdelta, eps=args.depth_eps)
                if v is not None:
                    L_depth = v
            elif args.depth_loss_type in ('ek_shift', 'ek_shift_constrained'):
                prob = _depth_prob_from_depthnet_out(x_depthnet, D, eps=args.depth_eps)
                if (prob is not None) and (Ek0_ref is not None) and (ent0_ref is not None):
                    constrained = (args.depth_loss_type == 'ek_shift_constrained')
                    L_depth = depth_loss_ek_shift_from_prob(
                        prob,
                        Ek0=Ek0_ref,
                        ent0=ent0_ref,
                        delta=args.depth_ek_delta,
                        constrained=constrained,
                        beta=args.depth_shape_beta,
                    )
            elif args.depth_loss_type == 'multi_peak':
                v = depth_loss_multi_peak_from_x(
                    x_depthnet, D,
                    K=args.depth_mpk_k,
                    steep=args.depth_mpk_steep,
                    margin=args.depth_mpk_margin,
                    sharp=args.depth_mpk_sharp,
                )
                if v is not None:
                    L_depth = v

        L_shape = torch.tensor(0.0, device=device)
        L_vertical = torch.tensor(0.0, device=device)
        L_div = torch.tensor(0.0, device=device)

        if task_head_holder is not None and (
            float(getattr(args, 'lambda_shape', 0.0)) > 0.0 or
            float(getattr(args, 'lambda_vertical', 0.0)) > 0.0 or
            float(getattr(args, 'lambda_div', 0.0)) > 0.0
        ):
            # try to read decode-related config
            pc_range = None
            voxel_size = None
            out_size_factor = None
            norm_bbox = True

            try:
                ch = find_center_head(model)
                coder = ch.bbox_coder
                pc_range = getattr(coder, 'pc_range', None)
                voxel_size = getattr(coder, 'voxel_size', None)
                out_size_factor = getattr(coder, 'out_size_factor', None)
                norm_bbox = getattr(ch, 'norm_bbox', True)
            except Exception:
                pass

            n_task_eff2 = 0

            for tid, out_d in task_head_holder.items():
                if not isinstance(out_d, dict):
                    continue
                hm_z = hm_logits.get(tid, None)
                if hm_z is None:
                    continue

                props = gather_topk_proposals_from_task(
                    out_d=out_d,
                    hm_z=hm_z,
                    topk=int(getattr(args, 'prop_topk', 200)),
                )

                boxes = decode_topk_props_simple(
                    props,
                    pc_range=pc_range,
                    voxel_size=voxel_size,
                    out_size_factor=out_size_factor,
                    norm_bbox=norm_bbox,
                    eps=float(getattr(args, 'geom_eps', 1e-6)),
                )

                if float(getattr(args, 'lambda_shape', 0.0)) > 0.0:
                    L_shape = L_shape + proposal_shape_range_loss_ascent(boxes, args)

                if float(getattr(args, 'lambda_vertical', 0.0)) > 0.0:
                    L_vertical = L_vertical + proposal_vertical_loss_ascent(boxes, args)

                if float(getattr(args, 'lambda_div', 0.0)) > 0.0:
                    L_div = L_div + proposal_diversity_loss_ascent(boxes, args)

                n_task_eff2 += 1

            n_task_eff2 = max(1, n_task_eff2)
            L_shape = L_shape / float(n_task_eff2)
            L_vertical = L_vertical / float(n_task_eff2)
            L_div = L_div / float(n_task_eff2)


        loss_ascent = (
            float(args.w_hm) * L_hm_pos
            + float(getattr(args, 'lambda_shape', 0.0)) * L_shape
            + float(getattr(args, 'lambda_vertical', 0.0)) * L_vertical
            + float(getattr(args, 'lambda_div', 0.0)) * L_div
            + float(getattr(args, 'lambda_area', 0.0)) * L_area
            + float(getattr(args, 'lambda_vol', 0.0))  * L_vol
            + float(getattr(args, 'lambda_reg', 0.0))  * L_reg
            + float(getattr(args, 'lambda_rot', 0.0))  * L_rot
            + float(args.lambda_depth) * L_depth
        )

        # ----------------------
        # Optional: print per-term gradient statistics wrt imgs (for lambda scaling)
        # ----------------------
        def _grad_stat(L_term, name):
            g = torch.autograd.grad(
                L_term, imgs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if g is None:
                print(f"[gradstat] {name}: grad=None")
                return None
            gd = g.detach()
            l2 = float(gd.norm(p=2).item())
            linf = float(gd.abs().max().item())
            meanabs = float(gd.abs().mean().item())
            print(f"[gradstat] {name}: L={float(L_term.detach().item()):.4e} |g|2={l2:.4e} |g|inf={linf:.4e} mean|g|={meanabs:.4e}")
            return gd

        def _cos(a, b):
            a = a.reshape(-1)
            b = b.reshape(-1)
            denom = (a.norm(p=2).item() * b.norm(p=2).item()) + 1e-12
            return float((a @ b).item() / denom)

        if getattr(args, 'print_grad_stat', False):
            if (not getattr(args, 'grad_stat_step0_only', False)) or (step == 0):
                # Note: these are gradients wrt imgs (pre pixel_weight masking).
                # This is still useful to understand relative scaling and direction conflicts.
                g_hm = _grad_stat(float(args.w_hm) * L_hm_pos, 'w_hm*L_hm')

                g_shape = None
                if float(getattr(args, 'lambda_shape', 0.0)) != 0.0:
                    g_shape = _grad_stat(
                        float(getattr(args, 'lambda_shape', 0.0)) * L_shape,
                        'lambda_shape*L_shape'
                    )

                g_vertical = None
                if float(getattr(args, 'lambda_vertical', 0.0)) != 0.0:
                    g_vertical = _grad_stat(
                        float(getattr(args, 'lambda_vertical', 0.0)) * L_vertical,
                        'lambda_vertical*L_vertical'
                    )

                g_div = None
                if float(getattr(args, 'lambda_div', 0.0)) != 0.0:
                    g_div = _grad_stat(
                        float(getattr(args, 'lambda_div', 0.0)) * L_div,
                        'lambda_div*L_div'
                    )

                g_area = None
                if float(getattr(args, 'lambda_area', 0.0)) != 0.0:
                    g_area = _grad_stat(
                        float(getattr(args, 'lambda_area', 0.0)) * L_area,
                        'lambda_area*L_area'
                    )

                g_vol = None
                if float(getattr(args, 'lambda_vol', 0.0)) != 0.0:
                    g_vol = _grad_stat(
                        float(getattr(args, 'lambda_vol', 0.0)) * L_vol,
                        'lambda_vol*L_vol'
                    )

                g_reg = None
                if float(getattr(args, 'lambda_reg', 0.0)) != 0.0:
                    g_reg = _grad_stat(
                        float(getattr(args, 'lambda_reg', 0.0)) * L_reg,
                        'lambda_reg*L_reg'
                    )

                g_rot = None
                if float(getattr(args, 'lambda_rot', 0.0)) != 0.0:
                    g_rot = _grad_stat(
                        float(getattr(args, 'lambda_rot', 0.0)) * L_rot,
                        'lambda_rot*L_rot'
                    )

                g_depth = None
                if float(getattr(args, 'lambda_depth', 0.0)) != 0.0:
                    g_depth = _grad_stat(
                        float(getattr(args, 'lambda_depth', 0.0)) * L_depth,
                        'lambda_depth*L_depth'
                    )

                # Direction agreement with HM (cosine)
                if g_hm is not None:
                    if g_shape is not None:
                        c = _cos(g_hm, g_shape)
                        cos_hist['shape'].append(c)
                        print(f"[gradstat] cos(hm,shape)={c:+.4f}")

                    if g_vertical is not None:
                        c = _cos(g_hm, g_vertical)
                        cos_hist['vertical'].append(c)
                        print(f"[gradstat] cos(hm,vertical)={c:+.4f}")

                    if g_div is not None:
                        c = _cos(g_hm, g_div)
                        cos_hist['div'].append(c)
                        print(f"[gradstat] cos(hm,div)={c:+.4f}")

                    if g_area is not None:
                        c = _cos(g_hm, g_area)
                        cos_hist['area'].append(c)
                        print(f"[gradstat] cos(hm,area)={c:+.4f}")

                    if g_vol is not None:
                        c = _cos(g_hm, g_vol)
                        cos_hist['vol'].append(c)
                        print(f"[gradstat] cos(hm,vol)={c:+.4f}")

                    if g_reg is not None:
                        c = _cos(g_hm, g_reg)
                        cos_hist['reg'].append(c)
                        print(f"[gradstat] cos(hm,reg)={c:+.4f}")

                    if g_rot is not None:
                        c = _cos(g_hm, g_rot)
                        cos_hist['rot'].append(c)
                        print(f"[gradstat] cos(hm,rot)={c:+.4f}")

                    if g_depth is not None:
                        c = _cos(g_hm, g_depth)
                        cos_hist['depth'].append(c)
                        print(f"[gradstat] cos(hm,depth)={c:+.4f}")

        model.zero_grad()

        if imgs.grad is not None:
            imgs.grad.zero_()
        loss_ascent.backward()

        # pixel_weight: cams + roi + patch
        pixel_weight = torch.ones((B, N_cam, 1, H, W), device=device)

        cam_mask = torch.zeros_like(pixel_weight)
        cam_mask[:, cam_ids, ...] = 1.0
        pixel_weight *= cam_mask

        if roi_mask is not None:
            pixel_weight *= roi_mask

        if args.roi_mask and args.roi_patch and (roi_mask is not None):
            patch_mask, _ = build_single_patch_mask(
                grads=imgs.grad,
                roi_mask=roi_mask,
                patch_size=args.patch_size,
                min_frac=args.patch_min_frac,
                min_size=args.patch_min_size,
                fallback=args.patch_fallback,
            )
            pixel_weight *= patch_mask

        with torch.no_grad():
            imgs.add_(args.alpha * imgs.grad.sign() * pixel_weight)
            diff = torch.clamp(imgs - imgs_base, min=-args.eps, max=args.eps)
            if roi_mask is not None:
                diff = diff * roi_mask
            imgs.copy_(imgs_base + diff)

    # --- summarize cosine similarities across ALL PGD steps ---
    if getattr(args, 'grad_stat_summary', False):
        def _summ(name, vals):
            if vals is None or len(vals) == 0:
                return
            arr = np.asarray(vals, dtype=np.float32)
            print(
                f"[gradstat][summary] {name}: n={arr.size} "
                f"mean={float(arr.mean()):+.4f} std={float(arr.std()):.4f} "
                f"min={float(arr.min()):+.4f} max={float(arr.max()):+.4f}"
            )

        _summ('cos(hm,shape)', cos_hist.get('shape'))
        _summ('cos(hm,vertical)', cos_hist.get('vertical'))
        _summ('cos(hm,div)', cos_hist.get('div'))
        _summ('cos(hm,area)', cos_hist.get('area'))
        _summ('cos(hm,vol)',  cos_hist.get('vol'))
        _summ('cos(hm,reg)',  cos_hist.get('reg'))
        _summ('cos(hm,rot)',  cos_hist.get('rot'))
        _summ('cos(hm,depth)', cos_hist.get('depth'))

    # write back final imgs (respect roi-mask-input output mode)
    imgs_final = imgs
    if args.roi_mask and args.roi_mask_input and (roi_mask is not None):
        fill_val = 0.0
        if args.roi_mask_output == 'masked':
            # Save what the model actually saw (masked outside ROI)
            imgs_final = imgs * roi_mask + fill_val * (1.0 - roi_mask)
        else:  # preserve
            # Keep outside-ROI pixels identical to original
            imgs_final = imgs * roi_mask + imgs_orig * (1.0 - roi_mask)

    inner = list(batch['img_inputs'][0])
    inner[0] = imgs_final.detach()
    img_inputs = list(batch['img_inputs'])
    img_inputs[0] = inner
    batch['img_inputs'] = img_inputs
    return batch
