"""Experiment lifecycle: setup, inference, attack, counting, and evaluation."""

import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel, scatter
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from .attack import attack_one_batch_phase1_hmonly
from .cli import parse_args
from .counting import find_center_head, install_counter, uninstall_counter
from .hooks import (
    _get_D_from_model,
    disable_checkpointing,
    register_all_task_head_dict_hooks,
    register_all_task_heatmap_logits_hooks,
    register_depthnet_hook,
)
from .timing import _CUDATimer

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        # Use raw model (no MMDataParallel) because PGD needs grads and we scatter manually
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()

        # IMPORTANT: PGD needs backward through the image backbone.
        # Some backbones (e.g., Swin) enable gradient checkpointing and may pass tuples,
        # which can crash torch.utils.checkpoint during backward.
        disable_checkpointing(model)
        cnt_state = None
        if args.count_pre_nms:
            center_head = find_center_head(model)
            cnt_state = install_counter(center_head, args.count_out)
            print("🧾 counting enabled:", args.count_out)

        # data_root / ann_file for ROI mask basename mapping
        data_root = None
        ann_file = None
        if isinstance(cfg.data.test, dict):
            data_root = cfg.data.test.get('data_root', None)
            ann_file = cfg.data.test.get('ann_file', None)
        elif isinstance(cfg.data.test, list) and len(cfg.data.test) > 0:
            data_root = cfg.data.test[0].get('data_root', None)
            ann_file = cfg.data.test[0].get('ann_file', None)

        infos_dict = None
        if ann_file is not None and os.path.isfile(ann_file):
            infos_obj = mmcv.load(ann_file)
            if isinstance(infos_obj, dict) and 'infos' in infos_obj:
                infos_dict = infos_obj

        # HM hooks (all tasks)
        hm_hooks, hm_task_holder = register_all_task_heatmap_logits_hooks(model, max_tasks=32)
        # Task head dict hooks (dim/reg/rot/etc.) for geometry losses
        th_hooks, task_head_holder = register_all_task_head_dict_hooks(model, max_tasks=32)
        D = None
        depth_hook = None
        depth_holder = None
        if args.lambda_depth > 0:
            D = _get_D_from_model(model)
            if D is None:
                raise RuntimeError('lambda_depth > 0 but cannot infer depth bins D')
            depth_hook, depth_holder, depth_name = register_depthnet_hook(model)
            print('✅ depth hook:', depth_name, '| D =', D)

        outputs = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        infer_ms_list = []

        for i, data in enumerate(data_loader):
            # sample_info for ROI masks
            sample_info = None
            if infos_dict is not None and i < len(infos_dict['infos']):
                sample_info = infos_dict['infos'][i]

            batch = scatter(data, [device.index if device.type == 'cuda' else 0])[0] if device.type == 'cuda' else data

            if args.attack:
                batch = attack_one_batch_phase1_hmonly(
                    model=model,
                    batch=batch,
                    args=args,
                    device=device,
                    hm_task_holder=hm_task_holder,
                    task_head_holder=task_head_holder,
                    sample_info=sample_info,
                    data_root=data_root,
                    D=D,
                    depth_holder=depth_holder,
                )

            # Inference only (keeps eval time tied to inference)
            with torch.no_grad():
                with _CUDATimer(enabled=(device.type == 'cuda')) as t_infer:
                    result = model(return_loss=False, rescale=True, **batch)

            infer_ms_list.append(t_infer.ms)
            print(f"[infer_only] i={i} ms={t_infer.ms:.3f}")

            outputs.extend(result)
            for _ in range(len(result)):
                prog_bar.update()

        if len(infer_ms_list) > 0:
            arr = np.array(infer_ms_list, dtype=np.float32)
            print(
                f"[infer_only summary] n={len(arr)} "
                f"total_ms={arr.sum():.3f} "
                f"avg_ms={arr.mean():.3f} "
                f"p50_ms={np.percentile(arr,50):.3f} "
                f"p95_ms={np.percentile(arr,95):.3f} "
                f"max_ms={arr.max():.3f}"
            )

        # cleanup hooks
        for h in hm_hooks:
            try:
                h.remove()
            except Exception:
                pass
        for h in th_hooks:
            try:
                h.remove()
            except Exception:
                pass
        if depth_hook is not None:
            try:
                depth_hook.remove()
            except Exception:
                pass
        if args.count_pre_nms and cnt_state is not None:
            center_head = find_center_head(model)
            uninstall_counter(center_head, cnt_state)
            print("🧾 counting closed")

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
