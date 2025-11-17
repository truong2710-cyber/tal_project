#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$1"

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

# --- default args for your inferencer ---
DEFAULT_ARGS=(
  --pose2d configs/body_2d_keypoint/rtmo/coco/rtmo-l_custom_coco.py
  --pose2d-weights pretrained_weights/epoch_240.pth
  --bbox-thr 0.5
)

# --- GPU / process setup (runs with torchrun if >1 GPU) ---
GPU_COUNT=$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)
echo "Detected $GPU_COUNT GPU(s)"

if (( GPU_COUNT > 1 )); then
  torchrun --standalone --nproc-per-node="$GPU_COUNT" demo/inferencer_demo_dist.py \
    "${img_dirs[@]}" \
    "${DEFAULT_ARGS[@]}" \
    "$@"
else
  # single-GPU / CPU
  export CUDA_VISIBLE_DEVICES=0
  python demo/inferencer_demo_dist.py \
    "${img_dirs[@]}" \
    "${DEFAULT_ARGS[@]}" \
    "$@"
fi

echo "✅ Done."
