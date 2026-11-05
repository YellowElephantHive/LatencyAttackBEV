# BEVDet thesis experiment jobs

The four repository-root `run_*_sbatch` files remain the stable submission
interface. They share this directory so paths, the container image, dataset,
checkpoint, and attack parameters cannot silently drift between experiments.

## Submit

```bash
sbatch run_baseline_sbatch
sbatch run_overload_sbatch
sbatch run_cam_count_sbatch
sbatch run_single_cam_position_sbatch
```

The four official jobs use the independent modular entrypoint
`tools/run_bevdet_experiment.py`. They retain the thesis GTX 1080 request for
reproducibility. Palmetto2 no longer advertises GTX 1080 nodes. A P100
command-line override was used on November 5, 2026 only to verify the
modular-code refactor, for example:

```bash
sbatch --gpus-per-node=p100:1 run_overload_sbatch
```

The P100 run is not part of the thesis or defense-slide experiments. Its timing
must not be compared directly with the GTX 1080 Ti thesis results.

The entrypoint uses the independent `tools/bevdet_experiment/` package and
does not import the local-only legacy implementation.

Before GPU smoke tests, submit the CPU-only modular unit-test job:

```bash
sbatch test_bevdet_modules_sbatch
```

It runs the CLI, camera/ROI, hook, heatmap, depth, geometry, and proposal tests
inside the thesis Apptainer image without requesting a GPU.

The names are historical: `run_overload_sbatch` currently runs the Gate-Sum
objective (`--hm-loss-mode gate_sum`), exactly as before this reorganization.

## Configuration

Edit `common.sh` once for a deliberate change shared by every experiment.
Values can also be overridden without editing files, which is useful for a
scratch checkout or a different cluster path:

```bash
BEVDET_HOST_ROOT=/scratch/$USER/BEVDet \
BEVDET_CONTAINER=/path/to/bevdet.sif \
sbatch --export=ALL run_baseline_sbatch
```

Camera matrices live in `camera_count.tsv` and `single_camera.tsv`. Keep their
tab-separated format and comment lines beginning with `#`.

## Safe checks

No GPU or container is needed to inspect the exact inner command:

```bash
SLURM_JOB_ID=DRYRUN BEVDET_DRY_RUN=1 \
BEVDET_HOST_ROOT="$PWD" bash run_overload_sbatch
```

Run `bash experiments/bevdet_attack/check.sh` before submitting. It validates
shell syntax, matrix contents, command counts, and every generated command
against `expected_commands.txt`, the golden snapshot of the official jobs.
