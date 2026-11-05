#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

jobs=(
  run_baseline_sbatch
  run_overload_sbatch
  run_cam_count_sbatch
  run_single_cam_position_sbatch
  test_bevdet_modules_sbatch
  experiments/bevdet_attack/common.sh
  experiments/bevdet_attack/lib.sh
)

bash -n "${jobs[@]}"

job_id=DRYRUN
export SLURM_JOB_ID="${job_id}"
export BEVDET_DRY_RUN=1
export BEVDET_HOST_ROOT="${repo_root}"

baseline_output="$(bash run_baseline_sbatch)"
overload_output="$(bash run_overload_sbatch)"
camera_count_output="$(bash run_cam_count_sbatch)"
single_camera_output="$(bash run_single_cam_position_sbatch)"

diff -u experiments/bevdet_attack/expected_commands.txt <(
  printf '%s\n' \
    "${baseline_output}" \
    "${overload_output}" \
    "${camera_count_output}" \
    "${single_camera_output}" \
    | grep '^cd /mmdetection3d && python '
)

[[ "$(grep -c '^cd /mmdetection3d && python ' <<<"${baseline_output}")" == 1 ]]
[[ "$(grep -c '^cd /mmdetection3d && python ' <<<"${overload_output}")" == 1 ]]
[[ "$(grep -c '^cd /mmdetection3d && python ' <<<"${camera_count_output}")" == 6 ]]
[[ "$(grep -c '^cd /mmdetection3d && python ' <<<"${single_camera_output}")" == 6 ]]
[[ "$(grep -c 'python tools/run_bevdet_experiment.py' <<<"${baseline_output}")" == 1 ]]
[[ "$(grep -c 'python tools/run_bevdet_experiment.py' <<<"${overload_output}")" == 1 ]]
[[ "$(grep -c 'python tools/run_bevdet_experiment.py' <<<"${camera_count_output}")" == 6 ]]
[[ "$(grep -c 'python tools/run_bevdet_experiment.py' <<<"${single_camera_output}")" == 6 ]]

[[ "${baseline_output}" != *" --attack "* ]]
[[ "${overload_output}" == *"--attack-cams all"* ]]
[[ "${overload_output}" == *"--hm-loss-mode gate_sum"* ]]
[[ "${overload_output}" == *"--eps 0.03137"* ]]
[[ "${overload_output}" == *"--alpha 0.003922"* ]]
[[ "${overload_output}" == *"--steps 20"* ]]

for cameras in 0 0,1 0,1,2 0,1,2,3 0,1,2,3,4 0,1,2,3,4,5; do
  printf -v quoted_cameras '%q' "${cameras}"
  grep -Fq -- "--attack-cams ${quoted_cameras}" <<<"${camera_count_output}"
done

for camera in 0 1 2 3 4 5; do
  grep -Fq -- "--attack-cams ${camera}" <<<"${single_camera_output}"
done

echo "BEVDet experiment job checks passed."
