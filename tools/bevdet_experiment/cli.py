"""Command-line interface for BEVDet thesis experiments."""

import argparse
import os
import warnings

from mmcv import DictAction

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # ----------------------
    # Integrated PGD attack (Phase 1: HM-only + ROI mask + ROI patch)
    # ----------------------
    parser.add_argument('--attack', action='store_true', help='Enable PGD attack before inference')

    # PGD params (normalized space)
    parser.add_argument('--eps', type=float, default=8/255.0)
    parser.add_argument('--alpha', type=float, default=2/255.0)
    parser.add_argument('--steps', type=int, default=5)

    # HM-only surrogate
    parser.add_argument('--w_hm', type=float, default=1.0)
    parser.add_argument('--hm-tau', type=float, default=0.1)
    parser.add_argument('--hm-steep', type=float, default=20.0)
    # HM surrogate variants (CenterPoint/BEVDet friendly)
    parser.add_argument('--hm-loss-mode',
                        choices=['gate_sum', 'overload_hardish', 'peak_count', 'topk_sum'],
                        default='gate_sum',
                        help=('HM surrogate type. gate_sum=area-like; overload_hardish=hard threshold-style (gate*prob); '
                              'peak_count=encourage more local maxima above tau; '
                              'topk_sum=maximize sum of topK probs per class.'))
    parser.add_argument('--hm-topk', type=int, default=200,
                        help='For hm-loss-mode=topk_sum: K for topK over spatial locations per class.')
    parser.add_argument('--hm-peak-kernel', type=int, default=3,
                        help='For hm-loss-mode=peak_count: maxpool kernel size (odd).')
    parser.add_argument('--hm-peak-steep', type=float, default=40.0,
                        help='For hm-loss-mode=peak_count: steepness for peakness sigmoid.')
    parser.add_argument('--hm-peak-eps', type=float, default=1e-3,
                        help='For hm-loss-mode=peak_count: small positive bias so exact maxima count as peaks.')

    # cameras to attack
    parser.add_argument('--attack-cams', type=str, default='all',
                        help='Comma-separated camera indices to attack, e.g. "0,1,2" or "all"')

    # ROI mask gating
    parser.add_argument('--roi-mask', action='store_true')
    parser.add_argument('--roi-mask-dir', type=str, default='mask')
    parser.add_argument('--roi-mask-thr', type=int, default=127)
    parser.add_argument('--roi-mask-missing', choices=['all','zero','error'], default='all')
    parser.add_argument('--roi-mask-input', action='store_true')
    parser.add_argument('--roi-mask-fill', choices=['zero','mean'], default='zero')
    parser.add_argument('--roi-mask-output', choices=['preserve','masked'], default='preserve',
                    help='When --roi-mask-input is on: preserve keeps outside-ROI pixels original; masked outputs the masked view')

    # ROI patch
    parser.add_argument('--roi-patch', action='store_true')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--patch-min-frac', type=float, default=0.6)
    parser.add_argument('--patch-min-size', type=int, default=8)
    parser.add_argument('--patch-fallback', choices=['shrink','skip','error'], default='shrink')

    # ----------------------
    # Depth attack (ported from attack_sa_ablation13)
    # ----------------------

    parser.add_argument('--lambda-depth', type=float, default=0.0,
                    help='Weight for depth auxiliary loss (0 disables depth loss)')
    parser.add_argument('--depth-loss-type',
                    choices=['entropy', 'kl_fixed', 'kl_shift', 'ek_shift', 'ek_shift_constrained', 'multi_peak'],
                    default='entropy')
    parser.add_argument('--depth-kstar', type=int, default=-1,
                    help='Target wrong bin for kl_fixed. -1 means D-1')
    parser.add_argument('--depth-kdelta', type=int, default=10,
                    help='Bin offset for kl_shift')
    parser.add_argument('--depth-ek-delta', type=float, default=5.0,
                    help='Delta in expected bin E[k] for ek_shift')
    parser.add_argument('--depth-shape-beta', type=float, default=1.0,
                    help='Penalty weight for ek_shift_constrained')
    parser.add_argument('--depth-eps', type=float, default=1e-12)

    # multi-peak
    parser.add_argument('--depth-mpk-k', type=float, default=2.0)
    parser.add_argument('--depth-mpk-steep', type=float, default=10.0)
    parser.add_argument('--depth-mpk-margin', type=float, default=0.0)
    parser.add_argument('--depth-mpk-sharp', type=float, default=0.0)

    # ----------------------
    # Geometry losses (optional): shrink dims / jitter reg / diversify rot
    # Controlled by lambda values (0 disables)
    # ----------------------
    parser.add_argument('--lambda-area', type=float, default=0.0,
                        help='Weight for L_area (shrink BEV box area via dim). 0 disables.')
    parser.add_argument('--lambda-reg', type=float, default=0.0,
                        help='Weight for L_reg (reg dispersion). 0 disables.')
    parser.add_argument('--lambda-rot', type=float, default=0.0,
                        help='Weight for L_rot (yaw dispersion proxy via rot head). 0 disables.')
    parser.add_argument('--lambda-vol', type=float, default=0.0,
                        help='Weight for L_vol (shrink 3D box volume via dim). 0 disables.')

    parser.add_argument('--area-dim-idx', type=str, default='0,1',
                        help='dim channels used as (w,l), default "0,1"')
    parser.add_argument('--vol-dim-idx', type=str, default='0,1,2',
                        help='dim channels used as (w,l,h) for L_vol, default "0,1,2"')
    parser.add_argument('--geom-eps', type=float, default=1e-6,
                        help='epsilon for geometry losses')

    # ----------------------
    # Proposal-level regularization
    # ----------------------
    parser.add_argument('--lambda-shape', type=float, default=0.0,
                        help='Weight for proposal-level shape range loss.')
    parser.add_argument('--lambda-vertical', type=float, default=0.0,
                        help='Weight for proposal-level vertical plausibility loss.')
    parser.add_argument('--lambda-div', type=float, default=0.0,
                        help='Weight for proposal diversity / overlap loss.')

    parser.add_argument('--prop-topk', type=int, default=200,
                        help='Top-K proposal locations per task used for proposal-level regularization.')

    parser.add_argument('--shape-lmin', type=float, default=1.0)
    parser.add_argument('--shape-lmax', type=float, default=5.0)
    parser.add_argument('--shape-wmin', type=float, default=0.5)
    parser.add_argument('--shape-wmax', type=float, default=3.0)
    parser.add_argument('--shape-hmin', type=float, default=0.5)
    parser.add_argument('--shape-hmax', type=float, default=3.0)

    parser.add_argument('--vert-zmin', type=float, default=-2.0)
    parser.add_argument('--vert-zmax', type=float, default=4.0)

    parser.add_argument('--div-min-dist', type=float, default=1.5,
                        help='Minimum desired BEV center distance between proposals.')
    parser.add_argument('--div-sample-k', type=int, default=64,
                        help='Subsample K proposals for pairwise diversity loss to avoid O(K^2) blowup.')


    parser.add_argument('--count-pre-nms', action='store_true',
                    help='Log candidate counts inside CenterHead postprocess to CSV')
    parser.add_argument('--count-out', type=str, default='work_dirs/cnt_candidates.csv',
                    help='CSV path for candidate counting')
    parser.add_argument('--print-grad-stat', action='store_true',
                    help='Print per-term dL/dimgs stats (norms and cosine) for debugging lambda scaling')
    parser.add_argument('--grad-stat-step0-only', action='store_true',
                    help='If set, only print grad stats at PGD step 0 for each batch')
    parser.add_argument('--grad-stat-summary', action='store_true',
                    help='If set, summarize cosine similarities across ALL PGD steps (mean/std/min/max) per batch')


    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args
