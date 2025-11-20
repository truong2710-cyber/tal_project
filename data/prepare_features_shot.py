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

# ------------------------------
# Filename -> frame index (for robustness if needed)
# ------------------------------
FNAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
def parse_frame_from_name(name: str) -> Tuple[str, int]:
    base = Path(name).name
    m = FNAME_RE.match(base)
    if not m:
        raise ValueError(f"Cannot parse frame from Name: {name}")
    return m.group("prefix"), int(m.group("idx"))

# ------------------------------
# Geometry / matching
# ------------------------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0

def expand_bbox_xyxy(b: np.ndarray, scale: float, img_wh: tuple[int,int] | None = None) -> np.ndarray:
    x1, y1, x2, y2 = map(float, b)
    cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
    w  = max(1e-6, x2 - x1); h  = max(1e-6, y2 - y1)
    w2 = 0.5 * w * scale; h2 = 0.5 * h * scale
    nx1, ny1, nx2, ny2 = cx - w2, cy - h2, cx + w2, cy + h2
    if img_wh is not None:
        W, H = img_wh
        nx1 = max(0.0, min(nx1, W-1)); ny1 = max(0.0, min(ny1, H-1))
        nx2 = max(0.0, min(nx2, W-1)); ny2 = max(0.0, min(ny2, H-1))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def extract_bbox_1x4(bbox_field) -> np.ndarray:
    arr = np.asarray(bbox_field, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (1, 4):
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.size != 4:
        return np.zeros((4,), dtype=np.float32)
    return arr

def match_person_bbox_strict_iou(
    pose_people: list[dict],
    track_bbox: np.ndarray,
    iou_thresh: float = 0.1,
    expand: float = 1.0,
    img_wh: tuple[int,int] | None = None,
) -> dict | None:
    if not pose_people:
        return None
    tb = track_bbox
    if expand and expand != 1.0:
        tb = expand_bbox_xyxy(track_bbox, expand, img_wh=img_wh)
    best_iou, best_idx = 0.0, -1
    for i, p in enumerate(pose_people):
        bbox = extract_bbox_1x4(p.get("bbox", [0,0,0,0]))
        iou = iou_xyxy(tb, bbox)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    if best_iou >= iou_thresh and best_idx >= 0:
        print(f"Found IoU match: {best_iou:.3f}")
        return pose_people[best_idx]
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
    w = max(1e-6, float(x2 - x1)); h = max(1e-6, float(y2 - y1))
    out = kp.copy()
    out[:, 0] = (out[:, 0] - x1) / w
    out[:, 1] = (out[:, 1] - y1) / h
    np.clip(out, 0.0, 1.0, out=out)
    return out

# ------------------------------
# Readers
# ------------------------------
def read_keypoints_json(path: Path) -> List[List[dict]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Keypoints JSON must be a list of frames. Got {type(data)} at {path}")
    return data

@dataclass
class TrackRecord:
    frame: int
    pid: int
    bbox: np.ndarray

def read_tracking_csv(tracking_csv: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """
    CSV headers:
      frame_number,Name,id,x1,y1,x2,y2,score,class_id
    Frame numbers start from 1 -> we shift to 0-based.
    Returns {pid: {frame_idx_0based: bbox}}
    """
    rows: List[TrackRecord] = []
    with tracking_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"frame_number", "id", "x1", "y1", "x2", "y2"}
        if not reader.fieldnames or not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"Tracking CSV missing headers in {tracking_csv}. Found: {reader.fieldnames}")
        for row in reader:
            try:
                fr1 = int(float(row["frame_number"]))   # starts at 1
                pid = int(float(row["id"]))
                x1 = float(row["x1"]); y1 = float(row["y1"]); x2 = float(row["x2"]); y2 = float(row["y2"])
            except Exception:
                continue
            # shift to 0-based
            fr0 = fr1 - 1
            rows.append(TrackRecord(fr0, pid, np.array([x1, y1, x2, y2], dtype=np.float32)))

    pid_to_framebox: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for r in rows:
        pid_to_framebox[r.pid][r.frame] = r.bbox
    return pid_to_framebox

# ------------------------------
# Per-clip extraction (N, M, 2)
# ------------------------------
def extract_clip_pose_sequence(
    clip_start_abs: int,
    clip_end_abs: int,
    pid: int,
    pid_to_framebox: Dict[int, Dict[int, np.ndarray]],
    keypoints: List[List[dict]],
    iou_thresh: float,
    expand: float,
    img_wh: tuple[int,int] | None,
) -> np.ndarray:
    N = clip_end_abs - clip_start_abs + 1
    target_M = None
    seq: List[Optional[np.ndarray]] = []

    for abs_f in range(clip_start_abs, clip_end_abs + 1):
        people = keypoints[abs_f] if 0 <= abs_f < len(keypoints) else []
        track_bbox = pid_to_framebox.get(pid, {}).get(abs_f, None)
        if track_bbox is None or not people:
            seq.append(None); continue

        person = match_person_bbox_strict_iou(people, track_bbox, iou_thresh=iou_thresh, expand=expand, img_wh=img_wh)
        if person is None or "keypoints" not in person:
            seq.append(None); continue

        kp = np.asarray(person["keypoints"], dtype=np.float32)
        if kp.ndim != 2 or kp.shape[1] != 2:
            seq.append(None); continue

        kp = normalize_kp_by_bbox(kp, track_bbox)

        if target_M is None:
            target_M = kp.shape[0]
        if kp.shape[0] < target_M:
            pad = np.zeros((target_M - kp.shape[0], 2), dtype=np.float32)
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
# Process a single child video under a parent
# ------------------------------
def process_child_video(
    root: Path,
    parent: str,
    child: str,
    K: int,
    S: int,
    iou_thresh: float,
    expand: float,
    img_wh: tuple[int,int] | None,
) -> List[Path]:
    """
    TAL in:   /shot/label/tal/<parent>/ShotEvent/<child>/<pid>/{*.json, match.json}
    Track:    /shot/label/tracking/<parent>/ShotEvent/<child>.csv
    Keypts:   /shot/keypoints/<parent>/ShotEvent/<child>.json
    Save to:  /shot/features/<parent>/ShotEvent/<child>/<pid>/<clip>.npy  (W, K*M*2)
    """
    tal_child = root / "shot" / "label" / "tal" / parent / "ShotEvent" / child
    track_csv = root / "shot" / "label" / "tracking" / parent / "ShotEvent" / f"{child}.csv"
    kp_json   = root / "shot" / "keypoints" / parent / "ShotEvent" / f"{child}.json"
    out_base  = root / "shot" / "features" / parent / "ShotEvent" / child

    written: List[Path] = []
    if not tal_child.exists(): return written
    if not track_csv.exists(): 
        print(f"[WARN] missing tracking: {track_csv}"); 
        return written
    if not kp_json.exists():
        print(f"[WARN] missing keypoints: {kp_json}");
        return written

    pid_to_framebox = read_tracking_csv(track_csv)
    keypoints = read_keypoints_json(kp_json)

    for pid_dir in sorted(tal_child.iterdir()):
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

        out_pid = out_base / pid_str

        for clip_key, (start_abs, end_abs) in match_map.items():
            seq = extract_clip_pose_sequence(
                clip_start_abs=int(start_abs),
                clip_end_abs=int(end_abs),
                pid=pid,
                pid_to_framebox=pid_to_framebox,
                keypoints=keypoints,
                iou_thresh=iou_thresh,
                expand=expand,
                img_wh=img_wh,
            )  # (N, M, 2)

            if seq is None:
                continue
            
            N, M, _ = seq.shape
            windows = sliding_windows(N, K, S)
            if not windows:
                continue

            out_pid.mkdir(parents=True, exist_ok=True)

            W = len(windows)
            out = np.zeros((W, K * M * 2), dtype=np.float32)
            for w_i, (s, e) in enumerate(windows):
                out[w_i] = seq[s:e, :, :].reshape(K * M * 2)

            out_path = out_pid / f"{clip_key}.npy"
            np.save(out_path, out)
            written.append(out_path)

    return written

# ------------------------------
# Traverse all parents/children
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare bbox-normalized pose features (W, K*M*2) for ShotEvent hierarchy.")
    parser.add_argument("--root", type=str, default=".", help="Project root (contains '/shot/').")
    parser.add_argument("--K", type=int, default=16, help="Window size in frames.")
    parser.add_argument("--S", type=int, default=8, help="Stride in frames.")
    parser.add_argument("--iou_thr", type=float, default=0.3, help="Min IoU to accept a pose match.")
    parser.add_argument("--expand", type=float, default=1.0, help="Expand factor for tracking bbox before IoU (1.0 = none).")
    parser.add_argument("--imgW", type=int, default=1280, help="Image width for bbox clipping when expanding.")
    parser.add_argument("--imgH", type=int, default=720, help="Image height for bbox clipping when expanding.")
    args = parser.parse_args()

    root = Path(args.root)
    tal_root = root / "shot" / "label" / "tal"
    if not tal_root.exists():
        raise FileNotFoundError(f"TAL root not found: {tal_root}")

    total_written = 0
    parents = [d for d in sorted(tal_root.iterdir()) if d.is_dir()]
    print(f"Found {len(parents)} parent videos under {tal_root}")

    for parent_dir in parents:
        parent = parent_dir.name
        se = parent_dir / "ShotEvent"
        if not se.exists(): 
            continue
        children = [c for c in sorted(se.iterdir()) if c.is_dir()]
        for child_dir in children:
            child = child_dir.name
            written = process_child_video(
                root=root,
                parent=parent,
                child=child,
                K=args.K,
                S=args.S,
                iou_thresh=args.iou_thr,
                expand=args.expand,
                img_wh=(args.imgW, args.imgH),
            )
            print(f"[{parent}/ShotEvent/{child}] wrote {len(written)} features")
            total_written += len(written)

    print(f"Done. Total feature files: {total_written}")

if __name__ == "__main__":
    main()
