#!/usr/bin/env bash
set -euo pipefail

# 使用说明：
# 1. 后台启动：
#    bash /home/yihaoyu/docker/bag/MOFTransformer_CMT/run_try_alignn_with_pretrain.sh
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
SCRIPT="$ROOT/MOFTransformer/try.py"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/pids"
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/try_alignn_with_pretrain_${TIMESTAMP}.log"
PID_FILE="$PID_DIR/try_alignn_with_pretrain_${TIMESTAMP}.pid"

PYTHON_BIN="${PYTHON_BIN:-python}"
# 默认放到另一张 A6000，避开当前预训练占满的 GPU-f772...
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-2e81c395-ac38-bfde-adba-34f3d967f7af}"

ALIGNN_TRY_ROOT_DATASET="${ALIGNN_TRY_ROOT_DATASET:-/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/all_N2}"
ALIGNN_TRY_DOWNSTREAM="${ALIGNN_TRY_DOWNSTREAM:-N2}"
ALIGNN_TRY_LOG_DIR="${ALIGNN_TRY_LOG_DIR:-/home/yihaoyu/docker/bag/MOFTransformer_CMT/outputs/all_N2/logs_try_with_pretrain}"
ALIGNN_TRY_LOAD_PATH="${ALIGNN_TRY_LOAD_PATH:-/home/yihaoyu/docker/bag/MOFTransformer_CMT/outputs/hmof_pretrain_GGM_MPP_1pct/logs_alignn_pretrain_mpp_only/alignn_hmof_pretrain_1pct_mpp_only_seed0_from_best/version_0/checkpoints/best.ckpt}"
ALIGNN_TRY_MAX_EPOCHS="${ALIGNN_TRY_MAX_EPOCHS:-80}"
ALIGNN_TRY_BATCH_SIZE="${ALIGNN_TRY_BATCH_SIZE:-4}"
ALIGNN_TRY_LR="${ALIGNN_TRY_LR:-1e-4}"
ALIGNN_TRY_WEIGHT_DECAY="${ALIGNN_TRY_WEIGHT_DECAY:-1e-3}"
ALIGNN_TRY_DROP_RATE="${ALIGNN_TRY_DROP_RATE:-0.1}"
ALIGNN_TRY_PRECISION="${ALIGNN_TRY_PRECISION:-16-mixed}"
ALIGNN_TRY_IMG_SIZE="${ALIGNN_TRY_IMG_SIZE:-20}"
ALIGNN_TRY_MAX_GRAPH_LEN="${ALIGNN_TRY_MAX_GRAPH_LEN:-300}"
ALIGNN_TRY_HID_DIM="${ALIGNN_TRY_HID_DIM:-384}"
ALIGNN_TRY_NUM_LAYERS="${ALIGNN_TRY_NUM_LAYERS:-6}"
ALIGNN_TRY_ALIGNN_LAYERS="${ALIGNN_TRY_ALIGNN_LAYERS:-2}"
ALIGNN_TRY_GRAPH_DROPOUT="${ALIGNN_TRY_GRAPH_DROPOUT:-0.1}"
ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS="${ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS:-3}"

if [ -z "$ALIGNN_TRY_LOAD_PATH" ]; then
  echo "错误：ALIGNN_TRY_LOAD_PATH 不能为空，请手动指定你要用的预训练 ckpt。" >&2
  exit 1
fi

if command -v stdbuf >/dev/null 2>&1; then
  BUFFER_CMD=(stdbuf -oL -eL)
else
  BUFFER_CMD=()
fi

nohup setsid env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ALIGNN_TRY_ROOT_DATASET="$ALIGNN_TRY_ROOT_DATASET" \
  ALIGNN_TRY_DOWNSTREAM="$ALIGNN_TRY_DOWNSTREAM" \
  ALIGNN_TRY_LOG_DIR="$ALIGNN_TRY_LOG_DIR" \
  ALIGNN_TRY_LOAD_PATH="$ALIGNN_TRY_LOAD_PATH" \
  ALIGNN_TRY_MAX_EPOCHS="$ALIGNN_TRY_MAX_EPOCHS" \
  ALIGNN_TRY_BATCH_SIZE="$ALIGNN_TRY_BATCH_SIZE" \
  ALIGNN_TRY_LR="$ALIGNN_TRY_LR" \
  ALIGNN_TRY_WEIGHT_DECAY="$ALIGNN_TRY_WEIGHT_DECAY" \
  ALIGNN_TRY_DROP_RATE="$ALIGNN_TRY_DROP_RATE" \
  ALIGNN_TRY_PRECISION="$ALIGNN_TRY_PRECISION" \
  ALIGNN_TRY_IMG_SIZE="$ALIGNN_TRY_IMG_SIZE" \
  ALIGNN_TRY_MAX_GRAPH_LEN="$ALIGNN_TRY_MAX_GRAPH_LEN" \
  ALIGNN_TRY_HID_DIM="$ALIGNN_TRY_HID_DIM" \
  ALIGNN_TRY_NUM_LAYERS="$ALIGNN_TRY_NUM_LAYERS" \
  ALIGNN_TRY_ALIGNN_LAYERS="$ALIGNN_TRY_ALIGNN_LAYERS" \
  ALIGNN_TRY_GRAPH_DROPOUT="$ALIGNN_TRY_GRAPH_DROPOUT" \
  ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS="$ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS" \
  "${BUFFER_CMD[@]}" \
  "$PYTHON_BIN" -u "$SCRIPT" \
  >> "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "started ALIGNN downstream try.py (with pretrain ckpt) in background"
echo "pid: $PID"
echo "pid_file: $PID_FILE"
echo "root_dataset: $ALIGNN_TRY_ROOT_DATASET"
echo "load_path: $ALIGNN_TRY_LOAD_PATH"
echo "log_dir: $ALIGNN_TRY_LOG_DIR"
echo "learning_rate: $ALIGNN_TRY_LR"
echo "precision: $ALIGNN_TRY_PRECISION"
echo "hid_dim: $ALIGNN_TRY_HID_DIM"
echo "num_layers: $ALIGNN_TRY_NUM_LAYERS"
echo "alignn_layers: $ALIGNN_TRY_ALIGNN_LAYERS"
echo "img_size: $ALIGNN_TRY_IMG_SIZE"
echo "max_graph_len: $ALIGNN_TRY_MAX_GRAPH_LEN"
echo "freeze_backbone_epochs: $ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS"
echo "log: $LOG_FILE"
echo "follow log with:"
echo "tail -f $LOG_FILE"
