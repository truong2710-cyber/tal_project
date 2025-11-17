#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ACTION_NAME = "SHOT"
ACTION_ID = 0  # single class (SHOT)
LABEL_MAP = {ACTION_NAME: ACTION_ID}

# ------------------------------
# Rounding helpers
# ------------------------------

def round1(x: float) -> float:
    return float(f"{x:.1f}")

def round2(x: float) -> float:
    return float(f"{x:.2f}")

@dataclass
class Annotation:
    label: str
    segment: List[float]        # [start_sec, end_sec] relative to clip start (seconds, 1 decimal)
    segment_frames: List[float] # [start_frame, end_frame) relative to clip start (end exclusive)
    label_id: int

    def to_dict(self):
        return {
            "label": self.label,
            "segment": self.segment,
            "segment(frames)": self.segment_frames,
            "label_id": self.label_id,
        }

def safe_int(x: str) -> int:
    return int(float(x.strip()))

# ------------------------------
# Stats
# ------------------------------

@dataclass
class Stats:
    total_clips: int = 0
    total_frames: int = 0
    total_action_frames: int = 0
    total_action_segments: int = 0

    def add_clip(self, n_frames: int, anns: List[Annotation]):
        self.total_clips += 1
        self.total_frames += n_frames
        # count action frames from annotations' segment(frames) [start, end)
        seg_frames = 0
        for a in anns:
            s = int(round(a.segment_frames[0]))
            e = int(round(a.segment_frames[1]))
            seg_frames += max(0, e - s)
        self.total_action_frames += seg_frames
        self.total_action_segments += len(anns)

    def summary(self) -> str:
        if self.total_clips == 0:
            return "No clips saved; no statistics to report."
        avg_clip_len = self.total_frames / self.total_clips
        avg_action_len = (self.total_action_frames / self.total_action_segments) if self.total_action_segments > 0 else 0.0
        avg_action_ratio = (self.total_action_frames / self.total_frames) if self.total_frames > 0 else 0.0
        return (
            "\n===== Dataset Statistics (after cropping; saved clips only) =====\n"
            f"Total clips saved:           {self.total_clips}\n"
            f"Total frames (clips):        {self.total_frames}\n"
            f"Total action frames:         {self.total_action_frames}\n"
            f"Total action segments:       {self.total_action_segments}\n"
            f"Average clip length (frames): {avg_clip_len:.2f}\n"
            f"Average action length (frames): {avg_action_len:.2f}\n"
            f"Average action ratio:        {avg_action_ratio:.3f}\n"
            "=================================================================\n"
        )

# ------------------------------
# IO: tracking & temporal
# ------------------------------

def read_tracking_presence(tracking_csv: Path) -> Dict[int, List[int]]:
    """
    Reads tracking CSV with columns:
      frame_number, Name, id, x1, y1, x2, y2, score, class_id

    Returns:
      dict pid(int) -> sorted unique absolute frame indices (0-based).
    Auto-normalizes to 0-based if the file appears 1-based.
    """
    if not tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv}")

    presence = defaultdict(set)
    seen_frames = set()

    with tracking_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"frame_number", "id"}
        if not reader.fieldnames or not need.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Tracking CSV missing required headers in {tracking_csv}.\n"
                f"Found: {reader.fieldnames}\nNeeded at least: {sorted(need)}"
            )

        for row in reader:
            try:
                frame = int(float(row["frame_number"]))
                pid = int(float(row["id"]))
            except Exception:
                continue
            presence[pid].add(frame)
            seen_frames.add(frame)

    # auto-normalize to 0-based if looks 1-based
    if seen_frames and (0 not in seen_frames) and (min(seen_frames) >= 1):
        presence = {pid: {f - 1 for f in frames} for pid, frames in presence.items()}

    return {pid: sorted(frames) for pid, frames in presence.items()}

def read_temporal_labels(temporal_csv: Path, video_filename: str) -> Dict[int, List[Tuple[int, int]]]:
    """
    temporal csv format:
    File Name,Player Tracking ID,Action Start Frame,Action End Frame
    <video_name.mp4>,<pid>,<start>,<end>
    Returns: pid -> list of (start,end) inclusive, 0-based.
    Filters rows by File Name == video_filename.
    """
    if not temporal_csv.exists():
        return {}

    intervals = defaultdict(list)
    with temporal_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"File Name", "Player Tracking ID", "Action Start Frame", "Action End Frame"}
        if not reader.fieldnames or not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"Temporal CSV missing required headers: {temporal_csv}")

        for row in reader:
            if (row.get("File Name") or "").strip() != video_filename:
                continue
            pid = safe_int(row["Player Tracking ID"])
            s = safe_int(row["Action Start Frame"])
            e = safe_int(row["Action End Frame"])
            if e < s:
                s, e = e, s
            intervals[pid].append((s, e))

    for pid in intervals:
        intervals[pid].sort()
    return intervals

# ------------------------------
# Clip grouping (consecutive frames)
# ------------------------------

def group_consecutive(frames: List[int]) -> List[List[int]]:
    if not frames:
        return []
    frames = sorted(set(frames))
    clips = []
    cur = [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            clips.append(cur)
            cur = [f]
    clips.append(cur)
    return clips

# ------------------------------
# Random crop that keeps action span at 30–50% of cropped clip
# ------------------------------

def pick_random_crop_window_from_intervals(
    clip_frames: List[int],
    intervals: List[Tuple[int, int]],
    min_ratio: float = 0.3,
    max_ratio: float = 0.5,
) -> Tuple[int, int]:
    """
    clip_frames: consecutive absolute frame ids for a PID-clip.
    intervals: list of (s,e) inclusive absolute frame ranges for SHOT (for that PID).
    Returns (start_local, end_local) inclusive indices inside clip_frames.
    If the clip has **no action intersection**, return (0, len(clip_frames)-1) and let caller skip later.
    If the clip is **all action**, returns full clip (no crop).
    """
    N = len(clip_frames)
    if N == 0:
        return 0, -1

    clip_start = clip_frames[0]
    clip_end = clip_frames[-1]

    # Build set of local indices that are SHOT inside this clip
    action_locals: List[int] = []
    for (s, e) in intervals:
        is_s = max(s, clip_start)
        is_e = min(e, clip_end)
        if is_e < is_s:
            continue
        for abs_f in range(is_s, is_e + 1):
            action_locals.append(abs_f - clip_start)

    if not action_locals:
        return 0, N - 1

    a_start = min(action_locals)
    a_end = max(action_locals)
    action_len = a_end - a_start + 1

    if action_len >= N:
        return 0, N - 1

    r = random.uniform(min_ratio, max_ratio)
    crop_len = int(math.ceil(action_len / r))
    crop_len = min(crop_len, N)

    min_start = max(0, a_end - crop_len + 1)
    max_start = min(a_start, N - crop_len)
    if min_start > max_start:
        start_local = max(0, min(a_start, N - crop_len))
    else:
        start_local = random.randint(min_start, max_start)
    end_local = start_local + crop_len - 1
    return start_local, end_local

# ------------------------------
# Emit TAL for one video (returns stats increment)
# ------------------------------

def emit_for_video(
    root: Path,
    match_name: str,
    video_name: str,
    fps: float,
    subset: str,
    stats: Stats,
    min_ratio: float = 0.3,
    max_ratio: float = 0.5,
) -> List[Path]:
    """
    Process one video and write TAL jsons:
      input video:   shot/raw/<match_name>/ShotEvent/<video_name>.mp4  (not read here, just for layout)
      tracking csv:  shot/label/tracking/<match_name>/ShotEvent/<video_name>.csv
      temporal csv:  shot/label/temporal/<video_name>.csv
      outputs:       shot/label/tal/<match_name>/ShotEvent/<video_name>/<pid>/<video_name>_<pid>_<clipid>.json
                     shot/label/tal/<match_name>/ShotEvent/<video_name>/<pid>/match.json
    Also updates stats (after cropping; saved clips only).
    """
    written: List[Path] = []

    tracking_csv = root / "shot" / "label" / "tracking" / match_name / "ShotEvent" / f"{video_name}.csv"
    temporal_csv = root / "shot" / "label" / "temporal" / f"{video_name}.csv"

    presence = read_tracking_presence(tracking_csv)  # pid -> absolute frames
    temporal = read_temporal_labels(temporal_csv, f"{video_name}.mp4")  # pid -> [(s,e)]

    out_base = root / "shot" / "label" / "tal" / match_name / "ShotEvent" / video_name
    out_base.mkdir(parents=True, exist_ok=True)

    for pid, frames in presence.items():
        clips = group_consecutive(frames)
        pid_dir = out_base / str(pid)
        matches: Dict[str, List[int]] = {}

        pid_intervals = temporal.get(pid, [])

        for clipid, clip_frames in enumerate(clips):
            if not clip_frames:
                continue

            # --- Random crop based on SHOT coverage (if any) ---
            crop_s_local, crop_e_local = pick_random_crop_window_from_intervals(
                clip_frames, pid_intervals, min_ratio=min_ratio, max_ratio=max_ratio
            )
            cropped_frames = clip_frames[crop_s_local:crop_e_local + 1]
            if not cropped_frames:
                continue

            # Build SHOT annotations by intersecting temporal intervals with CROPPED clip
            anns: List[Annotation] = []
            c_start = cropped_frames[0]
            c_end = cropped_frames[-1]

            for (s, e) in pid_intervals:
                is_s = max(s, c_start)
                is_e = min(e, c_end)
                if is_e < is_s:
                    continue
                rel_s = is_s - c_start
                rel_e_ex = (is_e - c_start) + 1
                seg_s = round1(rel_s / fps)
                seg_e = round1(rel_e_ex / fps)
                anns.append(
                    Annotation(
                        label=ACTION_NAME,
                        segment=[seg_s, seg_e],
                        segment_frames=[float(rel_s), float(rel_e_ex)],
                        label_id=ACTION_ID,
                    )
                )

            # ---- Skip clips with no SHOT after cropping ----
            if not anns:
                continue

            # Update stats (saved clip only)
            stats.add_clip(n_frames=len(cropped_frames), anns=anns)

            # Write clip json
            clip_key = f"{video_name}_{pid}_{clipid}"
            pid_dir.mkdir(parents=True, exist_ok=True)
            duration_sec = round2(len(cropped_frames) / fps)
            tal_obj = {
                clip_key: {
                    "subset": subset,
                    "duration": duration_sec,
                    "fps": fps,
                    "annotations": [a.to_dict() for a in anns],
                }
            }
            out_path = pid_dir / f"{clip_key}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(tal_obj, f, ensure_ascii=False, indent=2)
            written.append(out_path)

            # match.json uses absolute 0-based start/end for the CROPPED clip
            matches[clip_key] = [c_start, c_end]

        # Write match.json if we saved any clips for this pid
        if matches:
            match_path = pid_dir / "match.json"
            with match_path.open("w", encoding="utf-8") as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)
            written.append(match_path)

    return written

# ------------------------------
# Walk all videos
# ------------------------------

def main(root: str, fps: float, subset: str, seed: int | None, min_ratio: float, max_ratio: float):
    if seed is not None:
        random.seed(seed)

    rootp = Path(root)
    raw_root = rootp / "shot" / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    stats = Stats()
    all_written: List[Path] = []

    for match_dir in sorted(raw_root.iterdir()):
        if not match_dir.is_dir():
            continue
        match_name = match_dir.name
        se_dir = match_dir / "ShotEvent"
        if not se_dir.exists():
            continue
        for mp4 in sorted(se_dir.glob("*.mp4")):
            video_name = mp4.stem  # without .mp4
            try:
                written = emit_for_video(
                    root=rootp,
                    match_name=match_name,
                    video_name=video_name,
                    fps=fps,
                    subset=subset,
                    stats=stats,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                )
            except Exception as e:
                print(f"⚠️  Error processing {mp4}: {e}")
                continue
            all_written.extend(written)

    print(f"Done. Wrote {len(all_written)} files.")
    print(stats.summary())
    # for p in all_written[:10]:
    #     print("  ", p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TAL labels (SHOT) with random action-centered crops, plus stats.")
    parser.add_argument("--root", type=str, default=".", help="Project root (contains 'shot/').")
    parser.add_argument("--fps", type=float, default=60.0, help="Video FPS (default: 60).")
    parser.add_argument("--subset", type=str, default="Train", help="Subset name (default: 'Train').")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional, for reproducibility).")
    parser.add_argument("--min_ratio", type=float, default=0.3, help="Min action ratio inside cropped clip (default 0.3).")
    parser.add_argument("--max_ratio", type=float, default=0.5, help="Max action ratio inside cropped clip (default 0.5).")
    args = parser.parse_args()
    main(root=args.root, fps=args.fps, subset=args.subset, seed=args.seed, min_ratio=args.min_ratio, max_ratio=args.max_ratio)
