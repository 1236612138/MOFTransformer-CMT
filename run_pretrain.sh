#!/usr/bin/env bash
set -euo pipefail

# Unified ALIGNN/CMT pretraining entrypoint.
# Override settings with environment variables instead of creating new scripts.

ROOT="/home/yihaoyu/docker/bag/MOFTransformer_CMT"
SCRIPT="$ROOT/MOFTransformer/pretrain_alignn_hmof.py"
RUNTIME_LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/pids"
mkdir -p "$RUNTIME_LOG_DIR" "$PID_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/yihaoyu/anaconda3/envs/trans1/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${ALIGNN_PRETRAIN_SEED:-0}"
RUN_TAG="${RUN_TAG:-cmt_fixed}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RUNTIME_LOG_DIR/pretrain_${RUN_TAG}_seed${SEED}_${TIMESTAMP}.log"
PID_FILE="$PID_DIR/pretrain_${RUN_TAG}_seed${SEED}_${TIMESTAMP}.pid"

DATASET="${ALIGNN_PRETRAIN_ROOT_DATASET:-/home/yihaoyu/docker/bag/MOFTransformer_upgrade/MOFTransformer/hmof_pretrain_GGM_MPP_1pct}"
OUTPUT_DIR="${ALIGNN_PRETRAIN_LOG_DIR:-$ROOT/outputs/hmof_pretrain_GGM_MPP_1pct/pretrain_current}"
EXP_NAME="${ALIGNN_PRETRAIN_EXP_NAME:-alignn_hmof_${RUN_TAG}}"
CHEM_LABELS="${ALIGNN_PRETRAIN_CHEM_LABEL_PATH:-$ROOT/generated_labels/hmof_1pct_chem/chem_labels.csv}"
ZEOPP_LABELS="${ALIGNN_PRETRAIN_ZEOPP_LABEL_PATH:-$ROOT/generated_labels/hmof_1pct_zeopp/zeopp_labels.csv}"

GGM_WEIGHT="${ALIGNN_PRETRAIN_GGM_WEIGHT:-0.10}"
MPP_WEIGHT="${ALIGNN_PRETRAIN_MPP_WEIGHT:-1.00}"
CHEM_WEIGHT="${ALIGNN_PRETRAIN_CHEM_WEIGHT:-0.15}"
ZEOPP_WEIGHT="${ALIGNN_PRETRAIN_ZEOPP_WEIGHT:-0.30}"

for path in "$PYTHON_BIN" "$SCRIPT" "$DATASET/train.json" "$DATASET/val.json" "$DATASET/test.json" "$CHEM_LABELS"; do
  if [ ! -e "$path" ]; then
    echo "Error: required path does not exist: $path" >&2
    exit 1
  fi
done

if [ "$ZEOPP_WEIGHT" != "0" ] && [ "$ZEOPP_WEIGHT" != "0.0" ]; then
  if [ ! -f "$ZEOPP_LABELS" ]; then
    echo "Error: Zeo++ labels do not exist: $ZEOPP_LABELS" >&2
    exit 1
  fi
  # Use a real CSV parser because raw Zeo++ fields contain quoted newlines.
  if ! "$PYTHON_BIN" -c '
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    pairs = {(row["lcd"], row["pld"]) for row in csv.DictReader(handle)}
raise SystemExit(0 if len(pairs) > 1 else 1)
' "$ZEOPP_LABELS"; then
    echo "Error: Zeo++ LCD/PLD labels are constant and match the known parser bug." >&2
    echo "Reparse raw fields first, or temporarily set ALIGNN_PRETRAIN_ZEOPP_WEIGHT=0." >&2
    exit 1
  fi
fi

nohup setsid env \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.9" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ALIGNN_PRETRAIN_ROOT_DATASET="$DATASET" \
  ALIGNN_PRETRAIN_LOG_DIR="$OUTPUT_DIR" \
  ALIGNN_PRETRAIN_EXP_NAME="$EXP_NAME" \
  ALIGNN_PRETRAIN_SEED="$SEED" \
  ALIGNN_PRETRAIN_LOAD_PATH="${ALIGNN_PRETRAIN_LOAD_PATH:-}" \
  ALIGNN_PRETRAIN_MAX_EPOCHS="${ALIGNN_PRETRAIN_MAX_EPOCHS:-30}" \
  ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE="${ALIGNN_PRETRAIN_PER_GPU_BATCHSIZE:-1}" \
  ALIGNN_PRETRAIN_BATCH_SIZE="${ALIGNN_PRETRAIN_BATCH_SIZE:-4}" \
  ALIGNN_PRETRAIN_NUM_WORKERS="${ALIGNN_PRETRAIN_NUM_WORKERS:-4}" \
  ALIGNN_PRETRAIN_PRECISION="${ALIGNN_PRETRAIN_PRECISION:-16-mixed}" \
  ALIGNN_PRETRAIN_MAX_GRAPH_LEN="${ALIGNN_PRETRAIN_MAX_GRAPH_LEN:-260}" \
  ALIGNN_PRETRAIN_HID_DIM="${ALIGNN_PRETRAIN_HID_DIM:-384}" \
  ALIGNN_PRETRAIN_NUM_LAYERS="${ALIGNN_PRETRAIN_NUM_LAYERS:-8}" \
  ALIGNN_PRETRAIN_NUM_HEADS="${ALIGNN_PRETRAIN_NUM_HEADS:-8}" \
  ALIGNN_PRETRAIN_MLP_RATIO="${ALIGNN_PRETRAIN_MLP_RATIO:-4}" \
  ALIGNN_PRETRAIN_ALIGNN_LAYERS="${ALIGNN_PRETRAIN_ALIGNN_LAYERS:-2}" \
  ALIGNN_PRETRAIN_IMG_SIZE="${ALIGNN_PRETRAIN_IMG_SIZE:-25}" \
  ALIGNN_PRETRAIN_NBR_FEA_LEN="${ALIGNN_PRETRAIN_NBR_FEA_LEN:-64}" \
  ALIGNN_PRETRAIN_LR="${ALIGNN_PRETRAIN_LR:-6e-5}" \
  ALIGNN_PRETRAIN_WEIGHT_DECAY="${ALIGNN_PRETRAIN_WEIGHT_DECAY:-1e-2}" \
  ALIGNN_PRETRAIN_DROP_RATE="${ALIGNN_PRETRAIN_DROP_RATE:-0.1}" \
  ALIGNN_PRETRAIN_GRAPH_DROPOUT="${ALIGNN_PRETRAIN_GRAPH_DROPOUT:-0.1}" \
  ALIGNN_PRETRAIN_MPP_RATIO="${ALIGNN_PRETRAIN_MPP_RATIO:-0.20}" \
  ALIGNN_PRETRAIN_UNCERTAINTY_WEIGHTING="${ALIGNN_PRETRAIN_UNCERTAINTY_WEIGHTING:-0}" \
  ALIGNN_PRETRAIN_GGM_WEIGHT="$GGM_WEIGHT" \
  ALIGNN_PRETRAIN_MPP_WEIGHT="$MPP_WEIGHT" \
  ALIGNN_PRETRAIN_CHEM_WEIGHT="$CHEM_WEIGHT" \
  ALIGNN_PRETRAIN_ZEOPP_WEIGHT="$ZEOPP_WEIGHT" \
  ALIGNN_PRETRAIN_CHEM_LABEL_PATH="$CHEM_LABELS" \
  ALIGNN_PRETRAIN_ZEOPP_LABEL_PATH="$ZEOPP_LABELS" \
  ALIGNN_PRETRAIN_CHEM_LABEL_NAMES="${ALIGNN_PRETRAIN_CHEM_LABEL_NAMES:-density,metal_fraction,fraction_C,fraction_N,fraction_O}" \
  ALIGNN_PRETRAIN_ZEOPP_LABEL_NAMES="${ALIGNN_PRETRAIN_ZEOPP_LABEL_NAMES:-pld,lcd,av_fraction,asa_m2_g}" \
  "$PYTHON_BIN" -u "$SCRIPT" >> "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "Started pretraining"
echo "pid: $PID"
echo "pid_file: $PID_FILE"
echo "log: $LOG_FILE"
echo "output_dir: $OUTPUT_DIR"
echo "tasks: ggm=$GGM_WEIGHT mpp=$MPP_WEIGHT chem=$CHEM_WEIGHT zeopp=$ZEOPP_WEIGHT"
echo "follow: tail -f '$LOG_FILE'"
