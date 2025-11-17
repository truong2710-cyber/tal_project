#!/usr/bin/env bash
set -euo pipefail

# ---------------- predefined input sources ----------------
ROOT_DIR="../data/skating/raw"

# ---------------- knobs ----------------
K=${K:-2}                # number of parallel workers; set via:  K=6 bash run_parallel.sh
GPU_ID=${GPU_ID:-0}      # which GPU to use

# ---------------- default args ----------------
DEFAULT_ARGS=(
  --pose2d configs/body_2d_keypoint/rtmo/coco/rtmo-l_custom_coco.py
  --pose2d-weights pretrained_weights/epoch_240.pth
  # --show-progress
  --bbox-thr 0.5
)

# ---------------- collect inputs ----------------
# image patterns (case-insensitive)
IMG_GLOBS=( '*.jpg' '*.jpeg' '*.png' '*.bmp' '*.webp' '*.JPG' '*.JPEG' '*.PNG' '*.BMP' '*.WEBP' )

# --- collect leaf dirs that directly contain images ---
echo "Scanning leaf dirs under: $ROOT_DIR"
mapfile -d '' -t all_dirs < <(find "$ROOT_DIR" -type d -print0)

img_dirs=()
for dir in "${all_dirs[@]}"; do
  # skip if it has any subdirectories (i.e., not a leaf)
  if find "$dir" -mindepth 1 -type d -print -quit | grep -q .; then
    continue
  fi

  # check for at least one image file directly inside this dir
  has_img=0
  for pat in "${IMG_GLOBS[@]}"; do
    # shell globbing; needs nullglob to avoid literal pattern when no match
    shopt -s nullglob
    files=( "$dir"/$pat )
    shopt -u nullglob
    if (( ${#files[@]} > 0 )); then
      has_img=1
      break
    fi
  done

  (( has_img )) && img_dirs+=( "$dir" )
done

echo "Found ${#img_dirs[@]} leaf folders that directly contain images."
if (( ${#img_dirs[@]} == 0 )); then
  echo "No matching folders found. Double-check ROOT_DIR and file extensions."
  exit 1
fi

inputs=("${img_dirs[@]}")
N=${#inputs[@]}

# ---------------- split into K chunks ----------------
if (( K < 1 )); then K=1; fi
chunk_size=$(( (N + K - 1) / K ))

# ---------------- prepare logs folder ----------------
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Logs will be saved in $LOG_DIR (timestamp $timestamp)"

# ---------------- launch workers ----------------
echo "Launching $K parallel processes on GPU $GPU_ID (chunk size ≈ $chunk_size)"
pids=()
for (( i=0; i<K; i++ )); do
  start=$(( i * chunk_size ))
  len=$(( chunk_size ))
  if (( start + len > N )); then len=$(( N - start )); fi
  (( len > 0 )) || continue

  chunk=( "${inputs[@]:start:len}" )

  log_file="${LOG_DIR}/worker_${i}_${timestamp}.log"
  echo "  → Worker $i: ${#chunk[@]} items (indexes $start..$((start+len-1))) → $log_file"

  (
    set -e
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python demo/inferencer_demo.py \
      "${chunk[@]}" \
      "${DEFAULT_ARGS[@]}" \
      "$@"
  ) >"$log_file" 2>&1 &

  pids+=( $! )
done

# ---------------- wait for completion ----------------
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status == 0 )); then
  echo "✅ All inputs processed successfully."
else
  echo "⚠️ Some workers failed. Check logs in $LOG_DIR"
fi
