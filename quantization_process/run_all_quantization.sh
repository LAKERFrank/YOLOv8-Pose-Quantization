#!/bin/bash

# # for batch = 1
# env -u PYTHONPATH /tmp/ultra_export_venv/bin/python build_pose_onnx.py
# python3 build_pose_engine.py

# # for batch = 10
# env -u PYTHONPATH /tmp/ultra_export_venv/bin/python  build_pose_onnx_batch10.py
# python3 build_pose_engine_batch10.py

set -e

# ============================================================
# Config
# 只需要修改這裡
# ============================================================

export MODEL_PT="/workspaces/CameraSensor/Quantization/pose-dataset/weights/116-office-finetune/weights/best.pt"

export DATA_YAML="/workspaces/CameraSensor/Quantization/pose-dataset/datasets/office-calib/data.yaml"

# Optional common config
export IMGSZ=640
export DEVICE=0
export WS=8
export CALIB_FRAC=1
export SEED=0

echo "============================================================"
echo "[CONFIG]"
echo "MODEL_PT=${MODEL_PT}"
echo "DATA_YAML=${DATA_YAML}"
echo "IMGSZ=${IMGSZ}"
echo "DEVICE=${DEVICE}"
echo "============================================================"


# ============================================================
# batch = 1
# ============================================================

echo
echo "========== Building batch=1 =========="

export CALIB_BATCH=1

env -u PYTHONPATH \
    /tmp/ultra_export_venv/bin/python \
    build_pose_onnx.py

python3 build_pose_engine.py

# ============================================================
# batch = 10
# ============================================================

echo
echo "========== Building batch=10 =========="

export CALIB_BATCH=10

env -u PYTHONPATH \
    /tmp/ultra_export_venv/bin/python \
    build_pose_onnx_batch10.py

python3 build_pose_engine_batch10.py


echo
echo "============================================================"
echo "[DONE] All engines built."
echo "============================================================"
