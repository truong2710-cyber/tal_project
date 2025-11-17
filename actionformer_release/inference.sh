#!/bin/bash
set -euo pipefail

# Usage: ./run_pipeline.sh <frames_dir_or_video_file>
FRAME_INPUT=${1:-}
if [ -z "$FRAME_INPUT" ]; then
  echo "Usage: $0 <frames_dir_or_video_file>"
  exit 1
fi

# Activate conda in this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Helper: detect if input is a video file by extension (basic heuristic)
is_video_file() {
  local f="$1"
  [[ -f "$f" ]] && [[ "${f,,}" =~ \.(mp4|mov|avi|mkv|webm|m4v)$ ]]
}

# If input is a video → decode to a temp frames dir, else use as-is
TEMP_DIR=""
FRAME_DIR=""
if is_video_file "$FRAME_INPUT"; then
  echo "[INFO] Input is a video file. Decoding frames with ffmpeg..."
  TEMP_DIR="$(mktemp -d -t frames_XXXXXX)"
  # Ensure cleanup even if the script errors
  cleanup() {
    if [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]]; then
      echo "[INFO] Cleaning up temp frames dir: ${TEMP_DIR}"
      rm -rf "${TEMP_DIR}"
    fi
  }
  trap cleanup EXIT

  # Decode frames starting at 0 with 6-digit names
  ffmpeg -y -i "$FRAME_INPUT" -qscale:v 2 -start_number 0 "${TEMP_DIR}/%06d.jpg"
  FRAME_DIR="$TEMP_DIR"
  echo "[INFO] Frames written to ${FRAME_DIR}"
else
  # Treat input as a directory of frames
  if [[ ! -d "$FRAME_INPUT" ]]; then
    echo "[ERROR] '$FRAME_INPUT' is neither a video file nor a directory."
    exit 1
  fi
  FRAME_DIR="$FRAME_INPUT"
fi

# ===== Stage 1: Tracking & Pose (openmmlab env) =====
conda deactivate || true
conda activate openmmlab

mkdir -p ./output

# Tracking
python libs/utils/track_people.py \
  --frame_dir "${FRAME_DIR}" \
  --out_csv ./output/test_video.csv

# 2D Pose Estimation
python ../mmpose/demo/inferencer_demo.py \
  "${FRAME_DIR}" \
  --out_json ./output/test_video.json \
  --pose2d ../mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_custom_coco.py \
  --pose2d-weights ../mmpose/pretrained_weights/epoch_240.pth

# Prepare Features
python libs/utils/prepare_features.py \
  --tracking_csv ./output/test_video.csv \
  --keypoints_json ./output/test_video.json \
  --out_feature_dir ./output/test_video/

# ===== Stage 2: Inference (actionformer env) =====
conda deactivate
conda activate actionformer

python inference.py configs/shot_pose.yaml ckpt/shot_pose_test/ ./output/test_video/ \
  --out_video ./output/test_video_output.mp4 \
  --frame_dir "${FRAME_DIR}" \
  --csv_file ./output/test_video.csv \
  --score-thresh 0.5

# If we created a TEMP_DIR, it'll be removed by the trap on EXIT
echo "[DONE]"
