#!/usr/bin/env bash
set -euo pipefail

# 使用说明：
# 1. 后台启动：
#    bash /home/yihaoyu/docker/bag/MOFTransformer_CMT/run_pretrain_alignn_hmof_1pct.sh
#
# 2. 启动后请记住终端打印的：
#    - pid
#    - pid_file
#    - log
#
# 3. 检查任务是否仍在运行：
#    ps -fp $(cat /home/yihaoyu/docker/bag/MOFTransformer_CMT/pids/<pid文件名>.pid)
#
# 4. 重新连接服务器后继续查看日志：
#    tail -f /home/yihaoyu/docker/bag/MOFTransformer_CMT/logs/<日志文件名>.log
#
# 5. 停止任务：
#    kill $(cat /home/yihaoyu/docker/bag/MOFTransformer_CMT/pids/<pid文件名>.pid)

ROOT="/home/yihaoyu/docker/bag/MOFTransformer_CMT"
SCRIPT="$ROOT/MOFTransformer/pretrain_alignn_hmof.py"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/pids"
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/pretrain_alignn_hmof_1pct_${TIMESTAMP}.log"
PID_FILE="$PID_DIR/pretrain_alignn_hmof_1pct_${TIMESTAMP}.pid"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-f772dd3f-e279-8cdc-62dd-7b39740b2063}"

ALIGNN_PRETRAIN_ROOT_DATASET="${ALIGNN_PRETRAIN_ROOT_DATASET:-/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/hmof_pretrain_GGM_MPP_1pct}"
ALIGNN_PRETRAIN_LOG_DIR="${ALIGNN_PRETRAIN_LOG_DIR:-/home/yihaoyu/docker/bag/MOFTransformer_CMT/outputs/hmof_pretrain_GGM_MPP_1pct/logs_alignn_pretrain}"
ALIGNN_PRETRAIN_EXP_NAME="${ALIGNN_PRETRAIN_EXP_NAME:-alignn_hmof_pretrain_1pct}"
ALIGNN_PRETRAIN_MAX_EPOCHS="${ALIGNN_PRETRAIN_MAX_EPOCHS:-50}"
ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE="${ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE:-1}"
ALIGNN_PRETRAIN_BATCH_SIZE="${ALIGNN_PRETRAIN_BATCH_SIZE:-4}"
ALIGNN_PRETRAIN_NUM_WORKERS="${ALIGNN_PRETRAIN_NUM_WORKERS:-4}"
ALIGNN_PRETRAIN_PRECISION="${ALIGNN_PRETRAIN_PRECISION:-16-mixed}"
ALIGNN_PRETRAIN_HID_DIM="${ALIGNN_PRETRAIN_HID_DIM:-384}"
ALIGNN_PRETRAIN_NUM_LAYERS="${ALIGNN_PRETRAIN_NUM_LAYERS:-6}"
ALIGNN_PRETRAIN_ALIGNN_LAYERS="${ALIGNN_PRETRAIN_ALIGNN_LAYERS:-2}"
ALIGNN_PRETRAIN_IMG_SIZE="${ALIGNN_PRETRAIN_IMG_SIZE:-20}"

if command -v stdbuf >/dev/null 2>&1; then
  BUFFER_CMD=(stdbuf -oL -eL)
else
  BUFFER_CMD=()
fi

nohup setsid env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ALIGNN_PRETRAIN_ROOT_DATASET="$ALIGNN_PRETRAIN_ROOT_DATASET" \
  ALIGNN_PRETRAIN_LOG_DIR="$ALIGNN_PRETRAIN_LOG_DIR" \
  ALIGNN_PRETRAIN_EXP_NAME="$ALIGNN_PRETRAIN_EXP_NAME" \
  ALIGNN_PRETRAIN_MAX_EPOCHS="$ALIGNN_PRETRAIN_MAX_EPOCHS" \
  ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE="$ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE" \
  ALIGNN_PRETRAIN_BATCH_SIZE="$ALIGNN_PRETRAIN_BATCH_SIZE" \
  ALIGNN_PRETRAIN_NUM_WORKERS="$ALIGNN_PRETRAIN_NUM_WORKERS" \
  ALIGNN_PRETRAIN_PRECISION="$ALIGNN_PRETRAIN_PRECISION" \
  ALIGNN_PRETRAIN_HID_DIM="$ALIGNN_PRETRAIN_HID_DIM" \
  ALIGNN_PRETRAIN_NUM_LAYERS="$ALIGNN_PRETRAIN_NUM_LAYERS" \
  ALIGNN_PRETRAIN_ALIGNN_LAYERS="$ALIGNN_PRETRAIN_ALIGNN_LAYERS" \
  ALIGNN_PRETRAIN_IMG_SIZE="$ALIGNN_PRETRAIN_IMG_SIZE" \
  "${BUFFER_CMD[@]}" \
  "$PYTHON_BIN" -u "$SCRIPT" \
  >> "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "started ALIGNN hMOF pretraining (1pct) in background"
echo "pid: $PID"
echo "pid_file: $PID_FILE"
echo "root_dataset: $ALIGNN_PRETRAIN_ROOT_DATASET"
echo "log_dir: $ALIGNN_PRETRAIN_LOG_DIR"
echo "precision: $ALIGNN_PRETRAIN_PRECISION"
echo "hid_dim: $ALIGNN_PRETRAIN_HID_DIM"
echo "num_layers: $ALIGNN_PRETRAIN_NUM_LAYERS"
echo "alignn_layers: $ALIGNN_PRETRAIN_ALIGNN_LAYERS"
echo "img_size: $ALIGNN_PRETRAIN_IMG_SIZE"
echo "log: $LOG_FILE"
echo "follow log with:"
echo "tail -f $LOG_FILE"
