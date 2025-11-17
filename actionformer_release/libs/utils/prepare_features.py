#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_person_bbox_strict_iou(
    people: List[Dict[str, Any]], track_bbox: np.ndarray, iou_thr: float
) -> Optional[Dict[str, Any]]:
    """
    Among people (each has 'bbox' as 1x4 matrix), pick the highest-IoU bbox with track_bbox if IoU>=iou_thr.
    Returns that person dict or None.
    """
    best_iou = -1.0
    best = None
    for p in people:
        bbox = p.get("bbox", None)
        if bbox is None:
            continue
        bb = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bb.size != 4:
            continue
        iou = iou_xyxy(track_bbox, bb)
        if iou >= iou_thr and iou > best_iou:
            best_iou = iou
            best = p
            # print(f"Matched person with IoU={iou:.3f}")
    # if best is None:
    #     print("No match found.")
    return best


def normalize_kp_by_bbox(kp: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    out = np.empty_like(kp, dtype=np.float32)
    out[:, 0] = (kp[:, 0] - x1) / w
    out[:, 1] = (kp[:, 1] - y1) / h
    return np.clip(out, 0.0, 1.0)


def find_consecutive_clips(frame_idxs: List[int]) -> List[List[int]]:
    """Split sorted frame indices into consecutive clips (next == prev+1)."""
    if not frame_idxs:
        return []
    frames = sorted(set(frame_idxs))
    clips: List[List[int]] = []
    cur = [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            clips.append(cur)
            cur = [f]
    clips.append(cur)
    return clips


@dataclass
class ArgsCfg:
    tracking_csv: str
    keypoints_json: str
    out_feature_dir: str
    K: int
    S: int
    fps: float
    iou_thr: float


def load_tracking(
    tracking_csv: str,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, List[int]], int]:
    """
    Load tracking CSV: frame_number,Name,id,x1,y1,x2,y2
    Uses frame_number directly (assumed 0-based as in your example).
    Returns:
      pid_to_framebox: pid -> {abs_frame_idx0: bbox_xyxy}
      pid_to_frames:   pid -> sorted list of abs_frame_idx0
      min_frame_idx0:  global minimum 0-based frame index across CSV
    """
    pid_to_framebox: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    pid_to_frames: Dict[int, List[int]] = defaultdict(list)
    min_abs0 = math.inf

    with open(tracking_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["frame_number", "Name", "id", "x1", "y1", "x2", "y2"]
        if any(k not in reader.fieldnames for k in required):
            raise ValueError(f"CSV must have columns: {required}. Got: {reader.fieldnames}")
        for row in reader:
            f_abs0 = int(row["frame_number"])  # already 0-based
            pid = int(row["id"])
            x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])

            pid_to_framebox[pid][f_abs0] = np.array([x1, y1, x2, y2], dtype=np.float32)
            pid_to_frames[pid].append(f_abs0)
            if f_abs0 < min_abs0:
                min_abs0 = f_abs0

    if min_abs0 is math.inf:
        min_abs0 = 0
    # dedup + sort
    for pid, lst in pid_to_frames.items():
        pid_to_frames[pid] = sorted(set(lst))

    return pid_to_framebox, pid_to_frames, int(min_abs0)


def extract_clip_pose_sequence(
    clip_abs_frames: List[int],
    pid: int,
    pid_to_framebox: Dict[int, Dict[int, np.ndarray]],
    keypoints: List[List[Dict[str, Any]]],
    iou_thr: float,
) -> np.ndarray:
    """
    For a clip (list of absolute 0-based frame indices), returns array (N, M, 2),
    normalized per-frame by that pid's tracking bbox. If no match or missing → zeros.
    M chosen from first matched frame; if none matched → return None.
    """
    N = len(clip_abs_frames)
    seq: List[Optional[np.ndarray]] = [None] * N
    target_M: Optional[int] = None

    for i, abs_f in enumerate(clip_abs_frames):
        people = keypoints[abs_f] if 0 <= abs_f < len(keypoints) else []
        track_bbox = pid_to_framebox.get(pid, {}).get(abs_f, None)
        if track_bbox is None or not people:
            continue

        person = match_person_bbox_strict_iou(people, track_bbox, iou_thr)
        if person is None:
            continue
        kp = np.asarray(person.get("keypoints", []), dtype=np.float32)
        if kp.ndim != 2 or kp.shape[1] != 2 or kp.shape[0] == 0:
            continue

        kp = normalize_kp_by_bbox(kp, track_bbox)

        if target_M is None:
            target_M = kp.shape[0]
        # pad/truncate to target_M
        if kp.shape[0] < target_M:
            pad = np.zeros((target_M - kp.shape[0], 2), dtype=np.float32)
            kp = np.vstack([kp, pad])
        elif kp.shape[0] > target_M:
            kp = kp[:target_M, :]

        seq[i] = kp

    if target_M is None:
        return None

    out = np.zeros((N, target_M, 2), dtype=np.float32)
    for i in range(N):
        if seq[i] is not None:
            out[i] = seq[i]
    return out


def windows_from_clip(K: int, S: int, N: int) -> List[Tuple[int, int]]:
    """Generate [start, end) sliding windows over N frames."""
    if N < K:
        return []
    wins = []
    i = 0
    while i + K <= N:
        wins.append((i, i + K))
        i += S
    return wins


def build_features_for_clip(
    clip_abs_frames: List[int],
    pid: int,
    pid_to_framebox: Dict[int, Dict[int, np.ndarray]],
    keypoints: List[List[Dict[str, Any]]],
    iou_thr: float,
    K: int,
    S: int,
) -> np.ndarray:
    """
    Return (W, K*M*2) float32 for this clip. If no windows, returns shape (0, K*M*2) (with inferred M).
    """
    pose_seq = extract_clip_pose_sequence(
        clip_abs_frames, pid, pid_to_framebox, keypoints, iou_thr
    )
    if pose_seq is None:
        return None
    N, M, _ = pose_seq.shape
    win_pairs = windows_from_clip(K, S, N)
    if not win_pairs:
        return np.zeros((0, K * M * 2), dtype=np.float32)
    feats = np.zeros((len(win_pairs), K * M * 2), dtype=np.float32)
    for widx, (a, b) in enumerate(win_pairs):
        feats[widx] = pose_seq[a:b].reshape(-1)
    return feats


def main():
    pa = argparse.ArgumentParser(
        description="Extract (W, K*M*2) pose features per (pid, clip) from tracking CSV (with frame_number) and keypoints JSON."
    )
    pa.add_argument("--tracking_csv", required=True, type=str, help="CSV: frame_number,Name,id,x1,y1,x2,y2")
    pa.add_argument("--keypoints_json", required=True, type=str, help="Keypoints JSON: list[F] of list[P] dicts.")
    pa.add_argument("--out_feature_dir", required=True, type=str, help="Output directory for features.")
    pa.add_argument("--K", type=int, default=16, help="Window length (frames).")
    pa.add_argument("--S", type=int, default=8, help="Stride (frames).")
    pa.add_argument("--fps", type=float, default=60.0, help="FPS (kept for reference; not used in features).")
    pa.add_argument("--iou_thr", type=float, default=0.3, help="IoU threshold to match tracking bbox to keypoint bbox.")
    args = pa.parse_args()

    # Load tracking (now using frame_number directly; no filename parsing)
    pid_to_framebox, pid_to_frames, min_abs0 = load_tracking(args.tracking_csv)

    # Load keypoints JSON
    with open(args.keypoints_json, "r", encoding="utf-8") as f:
        keypoints = json.load(f)  # list over frames

    out_root = Path(args.out_feature_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # For each pid → clips → features + match.json (only if something saved)
    for pid, frames in pid_to_frames.items():
        clips = find_consecutive_clips(frames)

        pid_dir = out_root / str(pid)
        match_map: Dict[str, List[int]] = {}
        any_saved = False  # track whether we actually saved any .npy for this pid

        for clipid, clip_abs in enumerate(clips):
            if not clip_abs:
                continue
            feats = build_features_for_clip(
                clip_abs_frames=clip_abs,
                pid=pid,
                pid_to_framebox=pid_to_framebox,
                keypoints=keypoints,
                iou_thr=args.iou_thr,
                K=args.K,
                S=args.S,
            )
            if feats is None or feats.shape[0] == 0:
                continue

            # Create the pid folder lazily (only when we have something to save)
            if not any_saved:
                pid_dir.mkdir(parents=True, exist_ok=True)

            clip_name = f"{pid}_{clipid}"
            np.save(pid_dir / f"{clip_name}.npy", feats)
            any_saved = True

            # store absolute 0-based video frame indices (offset by global min for consistency)
            start_abs0 = int(min(clip_abs) - min_abs0)
            end_abs0 = int(max(clip_abs) - min_abs0)
            match_map[clip_name] = [start_abs0, end_abs0]

        # Only write match.json if we actually saved at least one feature file
        if any_saved:
            with (pid_dir / "match.json").open("w", encoding="utf-8") as f:
                json.dump(match_map, f, ensure_ascii=False, indent=2)


    print(f"Done. Features saved to: {out_root}")


if __name__ == "__main__":
    main()
