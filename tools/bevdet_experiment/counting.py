"""Pre-NMS candidate counting instrumentation."""

import csv
import os
import time
from types import MethodType

import torch

from mmdet3d.core import xywhr2xyxyr
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu

def find_center_head(model):
    if hasattr(model, 'pts_bbox_head'):
        return model.pts_bbox_head
    for _, m in model.named_modules():
        if m.__class__.__name__ == 'CenterHead':
            return m
    raise RuntimeError("CenterHead not found")

def install_counter(center_head, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    f = open(out_csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "sample_idx","task_id",
        "n_decode","n_pre_thr","n_after_thr","n_after_nms","n_after_range","n_final_merged",
        "ms_pre_nms","ms_nms","ms_post_nms"
    ])
    f.flush()

    orig_get_task_detections = center_head.get_task_detections
    orig_get_bboxes = center_head.get_bboxes
    center_head._cnt_cache = {}

    def patched_get_task_detections(self, num_class_with_bg, batch_cls_preds,
                                    batch_reg_preds, batch_cls_labels, img_metas, task_id):
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)
        else:
            post_center_range = None

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            sid = img_metas[i].get('sample_idx', i)
            key = (sid, task_id)
            d = self._cnt_cache.get(key, {})

            def _sync_cuda():
                if torch.cuda.is_available() and cls_preds.is_cuda:
                    torch.cuda.synchronize()

            # (a) pre-NMS timing: from entering this task to just before calling nms_gpu
            _sync_cuda()
            t0 = time.perf_counter()

            d["n_pre_thr"] = int(cls_preds.shape[0])
            d["ms_pre_nms"] = 0.0
            d["ms_nms"] = 0.0
            d["ms_post_nms"] = 0.0

            nms_rescale_factor = self.test_cfg.get(
                'nms_rescale_factor', [1.0 for _ in range(len(self.task_heads))]
            )[task_id]
            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels == cid, 3:6] = box_preds[cls_labels == cid, 3:6] * nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] * nms_rescale_factor

            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(cls_preds.shape[0], device=cls_preds.device, dtype=torch.long)
            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            top_scores_keep = None
            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor([self.test_cfg['score_threshold']], device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            d["n_after_thr"] = int(top_scores.shape[0])
            # end of pre-NMS prep (thresholding/topk etc.)
            _sync_cuda()
            t_pre = time.perf_counter()
            d["ms_pre_nms"] = (t_pre - t0) * 1000.0

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(
                    img_metas[i]['box_type_3d'](box_preds[:, :], self.bbox_coder.code_size).bev
                )
                nms_thresh = self.test_cfg['nms_thr'][task_id] if isinstance(self.test_cfg['nms_thr'], list) else self.test_cfg['nms_thr']

                # (b) NMS timing: only wrap nms_gpu
                _sync_cuda()
                t1 = time.perf_counter()
                selected = nms_gpu(
                    boxes_for_nms, top_scores,
                    thresh=nms_thresh,
                    pre_maxsize=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
                _sync_cuda()
                t2 = time.perf_counter()
                d["ms_nms"] = (t2 - t1) * 1000.0
                t_after_nms = t2
            else:
                selected = []
                d["ms_nms"] = 0.0
                _sync_cuda()
                t_after_nms = time.perf_counter()

            d["n_after_nms"] = int(selected.numel()) if isinstance(selected, torch.Tensor) else int(len(selected))

            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels == cid, 3:6] = box_preds[cls_labels == cid, 3:6] / nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / nms_rescale_factor

            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            if selected_boxes.shape[0] != 0:
                final_box_preds = selected_boxes
                final_scores = selected_scores
                final_labels = selected_labels
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = dict(bboxes=final_box_preds[mask], scores=final_scores[mask], labels=final_labels[mask])
                else:
                    predictions_dict = dict(bboxes=final_box_preds, scores=final_scores, labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size], dtype=dtype, device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0], dtype=top_labels.dtype, device=device))

            # (c) post-NMS timing: after nms_gpu returns -> predictions_dict ready
            _sync_cuda()
            t3 = time.perf_counter()
            d["ms_post_nms"] = (t3 - t_after_nms) * 1000.0

            d["n_after_range"] = int(predictions_dict["scores"].shape[0])
            d["n_decode"] = d["n_pre_thr"]
            self._cnt_cache[key] = d
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def patched_get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        ret_list = orig_get_bboxes(preds_dicts, img_metas, img=img, rescale=rescale)
        for i in range(len(ret_list)):
            sid = img_metas[i].get('sample_idx', i)
            n_final_merged = int(ret_list[i][1].shape[0])
            for task_id in range(len(preds_dicts)):
                d = self._cnt_cache.get((sid, task_id), {})
                writer.writerow([sid, task_id,
                                 d.get("n_decode",-1),
                                 d.get("n_pre_thr",-1),
                                 d.get("n_after_thr",-1),
                                 d.get("n_after_nms",-1),
                                 d.get("n_after_range",-1),
                                 n_final_merged,
                                 float(d.get("ms_pre_nms", 0.0)),
                                 float(d.get("ms_nms", 0.0)),
                                 float(d.get("ms_post_nms", 0.0))])
        f.flush()
        return ret_list

    center_head.get_task_detections = MethodType(patched_get_task_detections, center_head)
    center_head.get_bboxes = MethodType(patched_get_bboxes, center_head)

    return (f, orig_get_task_detections, orig_get_bboxes)

def uninstall_counter(center_head, state):
    if state is None:
        return
    f, orig_a, orig_b = state
    center_head.get_task_detections = orig_a
    center_head.get_bboxes = orig_b
    f.close()
