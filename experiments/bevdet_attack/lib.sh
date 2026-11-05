#!/usr/bin/env bash

_bevdet_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${_bevdet_dir}/common.sh"

bevdet_prepare_job() {
  cd "${BEVDET_HOST_ROOT}"
  mkdir -p logs work_dirs
}

_bevdet_shell_quote() {
  local quoted=() value
  for value in "$@"; do
    printf -v value '%q' "${value}"
    quoted+=("${value}")
  done
  local IFS=' '
  printf '%s' "${quoted[*]}"
}

# Usage: bevdet_run ATTACK_CAMS COUNT_CSV [clean|attack]
bevdet_run() {
  if (( $# != 3 )); then
    echo "bevdet_run requires ATTACK_CAMS, COUNT_CSV, and clean|attack" >&2
    return 2
  fi

  local attack_cams="$1"
  local count_csv="$2"
  local mode="$3"
  local -a test_args=(
    python "${BEVDET_ENTRYPOINT}"
    --config "${BEVDET_CONFIG}"
    --checkpoint "${BEVDET_CHECKPOINT}"
    --eval bbox
  )

  case "${mode}" in
    clean)
      ;;
    attack)
      test_args+=(
        --attack
        --eps "${BEVDET_EPS}"
        --alpha "${BEVDET_ALPHA}"
        --steps "${BEVDET_STEPS}"
        --w_hm "${BEVDET_W_HM}"
        --hm-loss-mode "${BEVDET_HM_LOSS_MODE}"
        --hm-tau "${BEVDET_HM_TAU}"
        --hm-steep "${BEVDET_HM_STEEP}"
        --attack-cams "${attack_cams}"
        --print-grad-stat
        --grad-stat-step0-only
      )
      ;;
    *)
      echo "Unknown experiment mode: ${mode}" >&2
      return 2
      ;;
  esac

  test_args+=(
    --count-pre-nms
    --count-out "${count_csv}"
    --cfg-options
    "data.test.data_root=${BEVDET_DATA_ROOT}"
    "data.test.ann_file=${BEVDET_ANN_FILE}"
  )

  local command
  command="cd $(_bevdet_shell_quote "${BEVDET_CONTAINER_ROOT}") && $(_bevdet_shell_quote "${test_args[@]}")"

  if [[ "${BEVDET_DRY_RUN:-0}" == 1 ]]; then
    printf '%s\n' "${command}"
    return 0
  fi

  apptainer exec --nv \
    --bind "${BEVDET_HOST_ROOT}:${BEVDET_CONTAINER_ROOT}" \
    "${BEVDET_CONTAINER}" \
    /bin/bash -lc "${command}"
}
