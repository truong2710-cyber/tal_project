#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm

# ------------------------------
# Filename → frame index parser
#  matches: <parent_video>_<frame>.jpg / .png
# ------------------------------
FNAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

def parse_frame_from_name(name: str) -> Tuple[str, int]:
    """
    Example: '2024-10-15_Minnesota_vs_St. Louis_000001.jpg' -> ('2024-10-15_Minnesota_vs_St. Louis', 1)
    """
    base = Path(name).name
    m = FNAME_RE.match(base)
    if not m:
        raise ValueError(f"Tracking 'Name' does not match '*_NNNNNN.ext': {name}")
    return m.group("prefix"), int(m.group("idx"))

# ------------------------------
# Geometry / matching utils
# ------------------------------
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
    aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1)) * max(0.0, (ay2 - ay1))
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0

def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = 0.5 * (ax1 + ax2); acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2); bcy = 0.5 * (by1 + by2)
    return float(np.hypot(acx - bcx, acy - bcy))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_int(x) -> int:
    return int(float(x))

# ------------------------------
# Tracking reader (from Name)
# ------------------------------
@dataclass
class TrackRecord:
    frame: int
    pid: int
    bbox: np.ndarray  # [x1,y1,x2,y2]

def read_tracking_from_name(tracking_csv: Path, expected_prefix: str) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Reads tracking CSV with headers:
      Name,id,x1,y1,x2,y2,score,class_id
    - Extracts frame index from Name suffix '_NNNNNN.jpg'
    - Keeps only rows whose Name prefix == expected_prefix (parent_video)
    - Auto-normalizes to 0-based if no 0 and min>=1
    Returns:
      {pid: {frame_idx_0based: bbox}}
    """
    if not tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv}")

    rows: List[TrackRecord] = []
    seen_frames = set()

    with tracking_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"Name", "id", "x1", "y1", "x2", "y2"}
        if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
            raise ValueError(f"Tracking CSV missing headers. Found: {reader.fieldnames}, need: {sorted(needed)}")

        for row in reader:
            name = (row.get("Name") or "").strip()
            try:
                prefix, frame_idx = parse_frame_from_name(name)
            except Exception:
                continue
            # filter to parent video
            if prefix != expected_prefix:
                continue

            try:
                pid = to_int(row["id"])
                x1 = float(row["x1"]); y1 = float(row["y1"]); x2 = float(row["x2"]); y2 = float(row["y2"])
            except Exception:
                continue

            rows.append(TrackRecord(frame=frame_idx, pid=pid, bbox=np.array([x1, y1, x2, y2], dtype=np.float32)))
            seen_frames.add(frame_idx)

    # 0-base normalize if looks 1-based
    shift = 1 if (seen_frames and 0 not in seen_frames and min(seen_frames) >= 1) else 0
    pid_to_framebox: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for r in rows:
        pid_to_framebox[r.pid][r.frame - shift] = r.bbox
    return pid_to_framebox

# ------------------------------
# Pose JSON reader
# ------------------------------
def read_keypoints_json(path: Path) -> List[List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Keypoints JSON must be a list of frames. Got {type(data)} at {path}")
    return data

# ------------------------------
# Match pose ↔ tracking
# ------------------------------
def match_person_bbox(pose_people: List[Dict[str, Any]], track_bbox: np.ndarray, iou_thr: float = 0.3) -> Optional[Dict[str, Any]]:
    if not pose_people:
        return None
    ious = []
    for p in pose_people:
        bbox = np.array(p.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
        ious.append(iou_xyxy(track_bbox, bbox[0]))
    best_iou = max(ious) if ious else 0.0
    if best_iou >= iou_thr:
        idx = int(np.argmax(ious))
        print(f"Found IoU match: {best_iou:.3f}")
        return pose_people[idx]
    print("No IoU match found")
    return None

# ------------------------------
# Windowing
# ------------------------------
def sliding_windows(length: int, K: int, S: int) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s + K <= length:
        out.append((s, s + K))
        s += S
    return out

# ------------------------------
# Normalization
# ------------------------------
def normalize_kp_by_bbox(kp: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1e-6, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    out = kp.copy()
    out[:, 0] = (out[:, 0] - x1) / w
    out[:, 1] = (out[:, 1] - y1) / h
    np.clip(out, 0.0, 1.0, out=out)
    return out

# ------------------------------
# Per-clip sequence (N, M, 2)
# ------------------------------
def extract_clip_pose_sequence(
    clip_start_abs: int,
    clip_end_abs: int,
    pid: int,
    pid_to_framebox: Dict[int, Dict[int, np.ndarray]],
    keypoints: List[List[Dict[str, Any]]],
    iou_thr: float = 0.3,
) -> np.ndarray:
    N = clip_end_abs - clip_start_abs + 1
    target_M = None
    seq: List[Optional[np.ndarray]] = []

    for abs_f in range(clip_start_abs, clip_end_abs + 1):
        people = keypoints[abs_f] if 0 <= abs_f < len(keypoints) else []
        track_bbox = pid_to_framebox.get(pid, {}).get(abs_f, None)

        if track_bbox is None or not people:
            seq.append(None); continue

        person = match_person_bbox(people, track_bbox, iou_thr)
        if person is None or "keypoints" not in person:
            seq.append(None); continue

        kp = np.asarray(person["keypoints"], dtype=np.float32)
        if kp.ndim != 2 or kp.shape[1] != 2:
            seq.append(None); continue

        kp = normalize_kp_by_bbox(kp, track_bbox)

        if target_M is None:
            target_M = kp.shape[0]
        if kp.shape[0] < target_M:
            pad = np.full((target_M - kp.shape[0], 2), 0.0, dtype=np.float32)
            kp = np.vstack([kp, pad])
        elif kp.shape[0] > target_M:
            kp = kp[:target_M, :]

        seq.append(kp)

    if target_M is None:
        return None
    
    out = np.zeros((N, target_M, 2), dtype=np.float32)
    for i in range(N):
        if seq[i] is not None:
            out[i] = seq[i]
    return out

# ------------------------------
# Process a single parent_video
# ------------------------------
def process_parent_video(
    root: Path,
    parent_video: str,
    K: int,
    S: int,
    iou_thr: float = 0.3
) -> List[Path]:
    """
    TAL:    skating/label/tal/<parent_video>/<pid>/{*.json, match.json}
    TRACK:  skating/label/temporal/<parent_video> exported.csv
    KEYPTS: skating/keypoints/<parent_video>.json
    Save:   skating/features/<parent_video>/<pid>/<clip>.npy  with shape (W, K*M*2)
    """
    tal_base = root / "skating" / "label" / "tal" / parent_video
    track_csv = root / "skating" / "label" / "temporal" / f"{parent_video} exported.csv"
    kp_json = root / "skating" / "keypoints" / f"{parent_video}.json"
    feat_base = root / "skating" / "features" / parent_video

    if not tal_base.exists():
        print(f"[WARN] TAL dir missing, skip: {tal_base}"); return []
    if not track_csv.exists():
        print(f"[WARN] Tracking missing, skip: {track_csv}"); return []
    if not kp_json.exists():
        print(f"[WARN] Keypoints missing, skip: {kp_json}"); return []

    # Use the parent_video as expected prefix for Name
    pid_to_framebox = read_tracking_from_name(track_csv, expected_prefix=parent_video)
    keypoints = read_keypoints_json(kp_json)
    written: List[Path] = []

    for pid_dir in sorted(tal_base.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid_str = pid_dir.name
        try:
            pid = int(pid_str)
        except Exception:
            continue

        match_path = pid_dir / "match.json"
        if not match_path.exists():
            continue
        with match_path.open("r", encoding="utf-8") as f:
            match_map = json.load(f)  # {clip_key: [start_abs, end_abs]}

        out_pid_dir = feat_base / pid_str
        ensure_dir(out_pid_dir)

        for clip_key, (start_abs, end_abs) in match_map.items():
            seq = extract_clip_pose_sequence(
                clip_start_abs=int(start_abs),
                clip_end_abs=int(end_abs),
                pid=pid,
                pid_to_framebox=pid_to_framebox,
                keypoints=keypoints,
                iou_thr=iou_thr
            )  # (N, M, 2)

            if seq is None:
                continue
            
            N, M, _ = seq.shape
            windows = sliding_windows(N, K, S)
            if not windows:
                continue

            W = len(windows)
            out = np.zeros((W, K * M * 2), dtype=np.float32)
            for w_i, (s, e) in enumerate(windows):
                out[w_i] = seq[s:e, :, :].reshape(K * M * 2)

            np.save(out_pid_dir / f"{clip_key}.npy", out)
            written.append(out_pid_dir / f"{clip_key}.npy")

    return written

# ------------------------------
# Walk all parent_videos
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Auto-prepare bbox-normalized pose features (W, K*M*2) for all TAL videos.")
    parser.add_argument("--root", type=str, default=".", help="Project root (contains 'skating/').")
    parser.add_argument("--K", type=int, default=16, help="Window size (frames).")
    parser.add_argument("--S", type=int, default=8, help="Stride (frames).")
    parser.add_argument("--iou_thr", type=float, default=0.3, help="IoU threshold for matching tracking bbox to pose.")
    args = parser.parse_args()

    root = Path(args.root)
    tal_root = root / "skating" / "label" / "tal"
    if not tal_root.exists():
        raise FileNotFoundError(f"TAL root not found: {tal_root}")

    parent_videos = [d.name for d in sorted(tal_root.iterdir()) if d.is_dir()]
    print(f"Found {len(parent_videos)} parent videos under {tal_root}")

    all_written: List[Path] = []
    for pv in tqdm(parent_videos):
        written = process_parent_video(root, pv, K=args.K, S=args.S, iou_thr=args.iou_thr)
        print(f"[{pv}] wrote {len(written)} features")
        all_written.extend(written)

    print(f"Done. Total feature files: {len(all_written)}")

if __name__ == "__main__":
    main()
