#!/bin/bash
set -euo pipefail

# Usage:
#   ./inference.sh <frames_dir_or_video_file> <af_config_yaml> <af_ckpt> [output_dir]
#
# Example:
#   ./inference.sh data/test_video/ configs/shot_pose.yaml ckpt/shot_pose_test/ output/test_run

FRAME_INPUT=${1:-}
AF_CONFIG=${2:-}
AF_CKPT=${3:-}
OUT_DIR=${4:-}

# === Default output folder if not provided ===
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="./output/"
fi
mkdir -p "${OUT_DIR}"

if [[ -z "${FRAME_INPUT}" || -z "${AF_CONFIG}" || -z "${AF_CKPT}" ]]; then
  echo "Usage: $0 <frames_dir_or_video_file> <af_config_yaml> <af_ckpt> [output_dir]"
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

# Paths inside OUT_DIR
TRACK_CSV="${OUT_DIR}/tracking.csv"
KEYPTS_JSON="${OUT_DIR}/keypoints.json"
FEAT_DIR="${OUT_DIR}/features"
OUT_VIDEO="${OUT_DIR}/inference.mp4"

# Tracking
python libs/utils/track_people.py \
  --frame_dir "${FRAME_DIR}" \
  --out_csv "${TRACK_CSV}"

# 2D Pose Estimation
python ../mmpose/demo/inferencer_demo.py \
  "${FRAME_DIR}" \
  --out_json "${KEYPTS_JSON}" \
  --pose2d ../mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_custom_coco.py \
  --pose2d-weights ../mmpose/pretrained_weights/epoch_240.pth

# Prepare Features
python libs/utils/prepare_features.py \
  --tracking_csv "${TRACK_CSV}" \
  --keypoints_json "${KEYPTS_JSON}" \
  --out_feature_dir "${FEAT_DIR}"

# ===== Stage 2: Inference (actionformer env) =====
conda deactivate
conda activate actionformer

python inference.py "${AF_CONFIG}" "${AF_CKPT}" "${FEAT_DIR}" \
  --out_video "${OUT_VIDEO}" \
  --frame_dir "${FRAME_DIR}" \
  --csv_file "${TRACK_CSV}" \
  --score-thresh 0.5

echo "[DONE] Outputs are in: ${OUT_DIR}"
