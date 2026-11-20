"""Generate clean/attack comparison frames without contaminating timing runs."""

import json
import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from bevdet_experiment.attack import attack_one_batch_phase1_hmonly
from bevdet_experiment.cli import parse_args
from bevdet_experiment.counting import (
    find_center_head,
    install_counter,
    uninstall_counter,
)
from bevdet_experiment.hooks import (
    disable_checkpointing,
    register_all_task_head_dict_hooks,
    register_all_task_heatmap_logits_hooks,
)
from bevdet_visualization.rendering import (
    compose_comparison_frame,
    make_camera_grid,
    make_difference_grid,
    render_bev_candidates,
)


class CandidateCollector:
    """Collect task-wise CPU snapshots emitted by the CenterHead counter."""

    def __init__(self):
        self.records = []

    def __call__(self, record):
        self.records.append(record)

    def merged(self):
        boxes = []
        scores = []
        labels = []
        selected = []
        offset = 0
        for record in self.records:
            task_boxes = record['boxes']
            task_scores = record['scores']
            task_labels = record['labels']
            boxes.append(task_boxes)
            scores.append(task_scores)
            labels.append(task_labels)
            selected.extend(
                offset + int(index) for index in record['selected'].reshape(-1))
            offset += int(task_boxes.shape[0])
        if not boxes:
            return {
                'boxes': torch.empty((0, 9)),
                'scores': torch.empty((0,)),
                'labels': torch.empty((0,), dtype=torch.long),
                'selected': torch.empty((0,), dtype=torch.long),
                'count': 0,
            }
        return {
            'boxes': torch.cat(boxes, dim=0),
            'scores': torch.cat(scores, dim=0),
            'labels': torch.cat(labels, dim=0),
            'selected': torch.as_tensor(selected, dtype=torch.long),
            'count': int(sum(record['boxes'].shape[0] for record in self.records)),
        }


def _add_visualization_args(parser):
    parser.description = 'Render a BEVDet clean/attack candidate comparison clip'
    parser.add_argument('--scene-name', default='scene-0103')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=40)
    parser.add_argument('--visualization-out',
                        default='work_dirs/visualization')
    parser.add_argument('--draw-limit', type=int, default=500)


def _prepare_test_cfg(cfg):
    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    return cfg, samples_per_gpu


def _clone_batch_images(batch):
    cloned = dict(batch)
    outer = list(batch['img_inputs'])
    inner = list(outer[0])
    inner[0] = inner[0].detach().clone()
    outer[0] = inner
    cloned['img_inputs'] = outer
    return cloned


def _batch_images(batch):
    image_inputs = batch['img_inputs']
    if hasattr(image_inputs, 'data'):
        image_inputs = image_inputs.data
    return image_inputs[0][0]


def _dataset_paths(cfg):
    test_cfg = cfg.data.test[0] if isinstance(cfg.data.test, list) \
        else cfg.data.test
    return test_cfg.get('data_root'), test_cfg.get('ann_file')


def _scene_tokens(data_root, scene_name):
    root = Path(data_root)
    versions = sorted(root.glob('v1.0-*/scene.json'))
    if not versions:
        raise FileNotFoundError(
            'could not find v1.0-*/scene.json under {}'.format(data_root))
    metadata = versions[0].parent
    scenes = json.loads((metadata / 'scene.json').read_text())
    samples = json.loads((metadata / 'sample.json').read_text())
    matching = [scene for scene in scenes if scene['name'] == scene_name]
    if not matching:
        raise ValueError('scene {} not found under {}'.format(
            scene_name, metadata))
    scene_token = matching[0]['token']
    return {sample['token'] for sample in samples
            if sample['scene_token'] == scene_token}


def _infer_with_candidates(model, batch, center_head, csv_path):
    collector = CandidateCollector()
    state = install_counter(
        center_head, str(csv_path), candidate_sink=collector)
    try:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **batch)
    finally:
        uninstall_counter(center_head, state)
    return result, collector.merged()


def main(argv=None):
    args = parse_args(argv=argv, configure_parser=_add_visualization_args)
    if not args.attack:
        raise ValueError('visualization requires --attack')
    if args.launcher != 'none':
        raise ValueError('visualization supports a single GPU only')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg, samples_per_gpu = _prepare_test_cfg(cfg)
    data_root, ann_file = _dataset_paths(cfg)
    if not data_root or not ann_file:
        raise ValueError('data.test.data_root and ann_file are required')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('a CUDA GPU is required to generate attack frames')
    model = model.to(device).eval()
    disable_checkpointing(model)

    hm_hooks, hm_holder = register_all_task_heatmap_logits_hooks(model)
    task_hooks, task_holder = register_all_task_head_dict_hooks(model)
    center_head = find_center_head(model)
    target_tokens = _scene_tokens(data_root, args.scene_name)
    target_indices = [
        index for index, info in enumerate(dataset.data_infos)
        if info['token'] in target_tokens
    ]
    target_indices.sort(key=lambda index: dataset.data_infos[index]['timestamp'])
    target_indices = target_indices[
        args.start_frame:args.start_frame + args.max_frames]
    target_set = set(target_indices)
    if not target_indices:
        raise ValueError('no dataset frames selected for {}'.format(
            args.scene_name))

    out_dir = Path(args.visualization_out)
    frame_dir = out_dir / 'frames'
    metric_dir = out_dir / 'counts'
    frame_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    try:
        rendered_index = 0
        for dataset_index, data in enumerate(data_loader):
            if dataset_index not in target_set:
                continue
            batch = scatter(data, [device.index])[0]
            clean_batch = _clone_batch_images(batch)
            attack_batch = _clone_batch_images(batch)
            clean_images = _batch_images(clean_batch).detach().clone()

            sample_info = dataset.data_infos[dataset_index]
            token = sample_info['token']
            _, clean_candidates = _infer_with_candidates(
                model, clean_batch, center_head,
                metric_dir / ('{:04d}_clean.csv'.format(rendered_index)))

            attack_batch = attack_one_batch_phase1_hmonly(
                model=model,
                batch=attack_batch,
                args=args,
                device=device,
                hm_task_holder=hm_holder,
                task_head_holder=task_holder,
                sample_info=sample_info,
                data_root=data_root,
                D=None,
                depth_holder=None,
            )
            attack_images = _batch_images(attack_batch).detach().clone()
            _, attack_candidates = _infer_with_candidates(
                model, attack_batch, center_head,
                metric_dir / ('{:04d}_attack.csv'.format(rendered_index)))

            gt_boxes = dataset.get_ann_info(dataset_index)[
                'gt_bboxes_3d'].tensor.cpu().numpy()
            clean_grid = make_camera_grid(clean_images)
            attack_grid = make_camera_grid(attack_images)
            difference_grid = make_difference_grid(
                clean_images, attack_images, amplification=8.0)
            clean_bev = render_bev_candidates(
                clean_candidates['boxes'], clean_candidates['scores'],
                clean_candidates['selected'], gt_boxes,
                draw_limit=args.draw_limit,
                candidate_color=(245, 170, 75),
            )
            attack_bev = render_bev_candidates(
                attack_candidates['boxes'], attack_candidates['scores'],
                attack_candidates['selected'], gt_boxes,
                draw_limit=args.draw_limit,
                candidate_color=(70, 95, 245),
            )
            composed = compose_comparison_frame(
                clean_grid, attack_grid, difference_grid,
                clean_bev, attack_bev,
                clean_candidates['count'], attack_candidates['count'],
                token, rendered_index, len(target_indices),
            )
            frame_path = frame_dir / ('frame_{:06d}.png'.format(rendered_index))
            if not cv2.imwrite(str(frame_path), composed):
                raise RuntimeError('failed to write {}'.format(frame_path))
            manifest.append({
                'frame': rendered_index,
                'dataset_index': dataset_index,
                'sample_token': token,
                'timestamp': int(sample_info['timestamp']),
                'clean_pre_nms_candidates': clean_candidates['count'],
                'attack_pre_nms_candidates': attack_candidates['count'],
            })
            print('[visualization] frame={} token={} clean={} attack={}'.format(
                rendered_index, token,
                clean_candidates['count'], attack_candidates['count']))
            rendered_index += 1
            if rendered_index >= len(target_indices):
                break
    finally:
        for hook in hm_hooks + task_hooks:
            try:
                hook.remove()
            except Exception:
                pass

    metadata = {
        'scene_name': args.scene_name,
        'attack': {
            'eps': args.eps,
            'alpha': args.alpha,
            'steps': args.steps,
            'hm_loss_mode': args.hm_loss_mode,
            'hm_tau': args.hm_tau,
            'hm_steep': args.hm_steep,
            'attack_cams': args.attack_cams,
        },
        'note': ('Visualization-only P100 run. Do not use its runtime as a '
                 'GTX 1080 Ti thesis timing result.'),
        'frames': manifest,
    }
    with (out_dir / 'manifest.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)
    print('Saved {} frames to {}'.format(len(manifest), frame_dir))
