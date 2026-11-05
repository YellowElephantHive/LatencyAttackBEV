#!/usr/bin/env bash

# Canonical environment and experiment parameters for the thesis runs.
# Override any value in the submitting environment when moving the project.
: "${BEVDET_HOST_ROOT:=/project/xiaoyon/trustai/lai/BEVDet}"
: "${BEVDET_CONTAINER:=/project/xiaoyon/trustai/lai/bevdet_apptainer.sif}"
: "${BEVDET_CONTAINER_ROOT:=/mmdetection3d}"
: "${BEVDET_ENTRYPOINT:=tools/run_bevdet_experiment.py}"
: "${BEVDET_CONFIG:=configs/bevdet/bevdet-sttiny.py}"
: "${BEVDET_CHECKPOINT:=checkpoint/bevdet-sttiny-pure.pth}"
: "${BEVDET_DATA_ROOT:=data/nuscenes_adv_patch}"
: "${BEVDET_ANN_FILE:=data/nuscenes_adv_patch/nuscenes_infos_val.pkl}"

# PGD / Gate-Sum parameters used by the reported experiments.
: "${BEVDET_EPS:=0.03137}"
: "${BEVDET_ALPHA:=0.003922}"
: "${BEVDET_STEPS:=20}"
: "${BEVDET_W_HM:=1.0}"
: "${BEVDET_HM_LOSS_MODE:=gate_sum}"
: "${BEVDET_HM_TAU:=0}"
: "${BEVDET_HM_STEEP:=20}"
