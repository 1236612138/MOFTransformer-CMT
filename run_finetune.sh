#!/usr/bin/env bash
set -euo pipefail

# Unified COF N2 finetuning entrypoint. Training automatically reloads the
# validation-best checkpoint and reports final test MAE, MSE and R2.

ROOT="/home/yihaoyu/docker/bag/MOFTransformer_CMT"
SCRIPT="$ROOT/MOFTransformer/try.py"
RUNTIME_LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/pids"
mkdir -p "$RUNTIME_LOG_DIR" "$PID_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/yihaoyu/anaconda3/envs/trans1/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${ALIGNN_TRY_SEED:-0}"
RUN_TAG="${RUN_TAG:-n2_stable}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RUNTIME_LOG_DIR/finetune_${RUN_TAG}_seed${SEED}_${TIMESTAMP}.log"
PID_FILE="$PID_DIR/finetune_${RUN_TAG}_seed${SEED}_${TIMESTAMP}.pid"

PRETRAIN_CKPT="${ALIGNN_TRY_LOAD_PATH:-}"
if [ -z "$PRETRAIN_CKPT" ] || [ ! -f "$PRETRAIN_CKPT" ]; then
  echo "Error: set ALIGNN_TRY_LOAD_PATH to an existing pretraining checkpoint." >&2
  echo "Example: ALIGNN_TRY_LOAD_PATH=/abs/path/best.ckpt bash $0" >&2
  exit 1
fi

nohup setsid env \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.9" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ALIGNN_TRY_EXP_NAME="${ALIGNN_TRY_EXP_NAME:-finetune_${RUN_TAG}}" \
  ALIGNN_TRY_SEED="$SEED" \
  ALIGNN_TRY_ROOT_DATASET="${ALIGNN_TRY_ROOT_DATASET:-/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/all_N2}" \
  ALIGNN_TRY_DOWNSTREAM="${ALIGNN_TRY_DOWNSTREAM:-N2}" \
  ALIGNN_TRY_LOG_DIR="${ALIGNN_TRY_LOG_DIR:-$ROOT/outputs/all_N2/finetune_current}" \
  ALIGNN_TRY_LOAD_PATH="$PRETRAIN_CKPT" \
  ALIGNN_TRY_MAX_EPOCHS="${ALIGNN_TRY_MAX_EPOCHS:-80}" \
  ALIGNN_TRY_BATCH_SIZE="${ALIGNN_TRY_BATCH_SIZE:-4}" \
  ALIGNN_TRY_LR="${ALIGNN_TRY_LR:-1e-5}" \
  ALIGNN_TRY_BACKBONE_LR_MULT="${ALIGNN_TRY_BACKBONE_LR_MULT:-0.5}" \
  ALIGNN_TRY_GRAPH_LR_MULT="${ALIGNN_TRY_GRAPH_LR_MULT:-1.0}" \
  ALIGNN_TRY_HEAD_LR_MULT="${ALIGNN_TRY_HEAD_LR_MULT:-5.0}" \
  ALIGNN_TRY_WEIGHT_DECAY="${ALIGNN_TRY_WEIGHT_DECAY:-1e-3}" \
  ALIGNN_TRY_DROP_RATE="${ALIGNN_TRY_DROP_RATE:-0.1}" \
  ALIGNN_TRY_PRECISION="${ALIGNN_TRY_PRECISION:-16-mixed}" \
  ALIGNN_TRY_WARMUP_STEPS="${ALIGNN_TRY_WARMUP_STEPS:-0.1}" \
  ALIGNN_TRY_DECAY_POWER="${ALIGNN_TRY_DECAY_POWER:-cosine}" \
  ALIGNN_TRY_GRAD_CLIP="${ALIGNN_TRY_GRAD_CLIP:-1.0}" \
  ALIGNN_TRY_EARLY_STOP_PATIENCE="${ALIGNN_TRY_EARLY_STOP_PATIENCE:-15}" \
  ALIGNN_TRY_IMG_SIZE="${ALIGNN_TRY_IMG_SIZE:-25}" \
  ALIGNN_TRY_MAX_GRAPH_LEN="${ALIGNN_TRY_MAX_GRAPH_LEN:-260}" \
  ALIGNN_TRY_HID_DIM="${ALIGNN_TRY_HID_DIM:-384}" \
  ALIGNN_TRY_NUM_LAYERS="${ALIGNN_TRY_NUM_LAYERS:-8}" \
  ALIGNN_TRY_NUM_HEADS="${ALIGNN_TRY_NUM_HEADS:-8}" \
  ALIGNN_TRY_ALIGNN_LAYERS="${ALIGNN_TRY_ALIGNN_LAYERS:-2}" \
  ALIGNN_TRY_GRAPH_DROPOUT="${ALIGNN_TRY_GRAPH_DROPOUT:-0.1}" \
  ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS="${ALIGNN_TRY_FREEZE_BACKBONE_EPOCHS:-4}" \
  ALIGNN_TRY_GRAPH_WARMUP_EPOCHS="${ALIGNN_TRY_GRAPH_WARMUP_EPOCHS:-3}" \
  ALIGNN_TRY_REGRESSION_HEAD="${ALIGNN_TRY_REGRESSION_HEAD:-linear}" \
  ALIGNN_TRY_REG_HEAD_DROPOUT="${ALIGNN_TRY_REG_HEAD_DROPOUT:-0.1}" \
  "$PYTHON_BIN" -u "$SCRIPT" >> "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "Started COF N2 finetuning and automatic best-checkpoint testing"
echo "pid: $PID"
echo "pid_file: $PID_FILE"
echo "seed: $SEED"
echo "pretrain_ckpt: $PRETRAIN_CKPT"
echo "log: $LOG_FILE"
echo "follow: tail -f '$LOG_FILE'"
