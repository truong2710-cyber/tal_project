# TAL label preparation script (with random crop + skip-empty)
# - Cleans "NOT_SURE" labels
# - Uses fixed LABEL_MAP (alphabetical mapping provided by you)
# - Finds per-person clips of consecutive frames
# - Randomly crops each clip so that the action span occupies 30–50% of the cropped clip
# - Skips saving clips that contain no actions
# - Emits one JSON per (saved) clip with TAL-style annotations
# - Emits a match.json per (video, pid) listing absolute cropped clip [start, end] in 0-based video frames
#
# Usage:
#   python prepare_tal_labels.py --frame_path .../frame/<video_root>/ --fps 60 --subset Train
#   (csv_path auto-derived below unless you pass a value and remove the override)
#
# Notes:
#   - Assumes video FPS = 60 by default (configurable via --fps).
#   - Frame indices are parsed from filenames that end with "_<frame_index>.jpg" (or .png).

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

LABEL_MAP = {
    "ACCEL_BACK": 0,
    "ACCEL_FORW": 1,
    "FACEOFF_BODY_POSITION": 2,
    "GLID_BACK": 3,
    "GLID_FORW": 4,
    "MAINTAIN_POSITION": 5,
    "ON_A_KNEE": 6,
    "POST_WHISTLE_GLIDING": 7,
    "PRONE": 8,
    "RAPID_DECELERATION": 9,
    "SHOT": 10,
    "TRANS_BACK_TO_FORW": 11,
    "TRANS_FORW_TO_BACK": 12,
}

# ------------------------------
# Helpers
# ------------------------------

FNAME_RE = re.compile(r"^(?P<base>.+)_(?P<idx>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

def parse_video_and_frame(filename: str) -> Tuple[str, int]:
    """
    Parse video name (everything before the _<frame> suffix) and integer frame index.
    Example: "2024-10-15_Minnesota_vs_St. Louis_011085.jpg" -> ("2024-10-15_Minnesota_vs_St. Louis", 11085)
    """
    name = os.path.basename(filename)
    m = FNAME_RE.match(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern '*_NNNNNN.ext': {filename}")
    base = m.group("base")
    idx = int(m.group("idx"))
    return base, idx

def replace_frame_with_label_tal(path: Path) -> Path:
    """
    Replace the 'frame' component in a path with 'label/tal'.
    If 'frame' isn't present, appends 'label/tal' at the end.
    """
    parts = list(path.parts)
    try:
        i = parts.index("frame")
        parts[i] = "label"
        parts.insert(i + 1, "tal")
        return Path(*parts)
    except ValueError:
        # 'frame' not found: default to sibling 'label/tal' under the same root
        return path / "label" / "tal"

def round1(x: float) -> float:
    return float(f"{x:.1f}")

def round2(x: float) -> float:
    return float(f"{x:.2f}")

@dataclass
class Annotation:
    label: str
    segment: List[float]        # [start_sec, end_sec] relative to clip start (seconds, 1 decimal)
    segment_frames: List[float] # [start_frame, end_frame] relative to clip start (frames; end is exclusive)
    label_id: int

    def to_dict(self):
        return {
            "label": self.label,
            "segment": self.segment,
            "segment(frames)": self.segment_frames,
            "label_id": self.label_id,
        }

# ------------------------------
# CSV processing
# ------------------------------

def load_and_process_csv(csv_path: str) -> List[dict]:
    """
    Load CSV with columns:
    filename, person_id, x1, y1, x2, y2, score, class_name, activity_label
    - Clears cells with activity containing 'NOT_SURE' (sets to '')
    Returns list of rows as dicts.
    """
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for raw in reader:
            if not raw or len(raw) < 9:
                continue
            filename = raw[0].strip()
            person_id = raw[1].strip()
            # bbox and score
            x1, y1, x2, y2 = raw[2].strip(), raw[3].strip(), raw[4].strip(), raw[5].strip()
            score = raw[6].strip()
            cls = raw[7].strip()
            activity = raw[8].strip() if len(raw) > 8 else ""

            # Clear NOT_SURE
            if "NOT_SURE" in activity.upper():
                activity = ""

            video_name, frame_idx = parse_video_and_frame(filename)

            rows.append({
                "filename": filename,
                "video_name": video_name,
                "frame_idx": frame_idx,
                "person_id": person_id,
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "class": cls,
                "activity": activity,
            })
    return rows

# ------------------------------
# Clip grouping & mappings
# ------------------------------

def group_person_clips(rows: List[dict]) -> Dict[Tuple[str, str], List[List[int]]]:
    """
    For each (video_name, person_id), find clips of consecutive frames containing that person.
    Returns map to a list of clips, each clip is a sorted list of absolute frame indices.
    """
    by_key = defaultdict(list)  # (video, pid) -> list of frame_idx
    for r in rows:
        by_key[(r["video_name"], r["person_id"])].append(r["frame_idx"])

    clips_by_key = {}
    for key, frames in by_key.items():
        frames = sorted(set(frames))
        clips = []
        if not frames:
            clips_by_key[key] = clips
            continue
        current = [frames[0]]
        for f in frames[1:]:
            if f == current[-1] + 1:
                current.append(f)
            else:
                clips.append(current)
                current = [f]
        clips.append(current)
        clips_by_key[key] = clips
    return clips_by_key

def build_frame_to_activity(rows: List[dict]) -> Dict[Tuple[str, str], Dict[int, str]]:
    """
    For each (video_name, person_id), map frame_idx -> activity (may be '')
    Prefer non-empty activity if duplicates exist.
    """
    mapping = defaultdict(dict)
    for r in rows:
        key = (r["video_name"], r["person_id"])
        f = r["frame_idx"]
        act = r["activity"]
        if f not in mapping[key] or (act and not mapping[key].get(f)):
            mapping[key][f] = act
        else:
            mapping[key].setdefault(f, act)
    return mapping

# ------------------------------
# Random crop helper
# ------------------------------

def pick_random_crop_window(
    frames: List[int],
    f2a: Dict[int, str],
    min_ratio: float = 0.3,
    max_ratio: float = 0.5,
) -> Tuple[int, int]:
    """
    Given a consecutive clip 'frames' (absolute frame ids) and per-frame activity mapping f2a,
    choose a random crop [start_local, end_local] (clip-local indices, inclusive) such that
    the minimal action span occupies a ratio in [min_ratio, max_ratio] of the cropped clip.
    If no action exists, returns the full clip.
    """
    N = len(frames)
    if N == 0:
        return 0, -1

    # indices (clip-local) where there is action
    action_locals = [i for i, f in enumerate(frames) if f2a.get(f, "")]
    if not action_locals:
        return 0, N - 1  # no action → no crop

    a_start = action_locals[0]
    a_end = action_locals[-1]
    action_len = a_end - a_start + 1

    r = random.uniform(min_ratio, max_ratio)
    crop_len = int(math.ceil(action_len / r))
    crop_len = min(crop_len, N)

    # place the window so it fully contains the action span
    min_start = max(0, a_end - crop_len + 1)
    max_start = min(a_start, N - crop_len)
    if min_start > max_start:
        start_local = max(0, min(a_start, N - crop_len))
    else:
        start_local = random.randint(min_start, max_start)
    end_local = start_local + crop_len - 1
    return start_local, end_local

# ------------------------------
# Emission
# ------------------------------

def emit_clip_jsons_and_matches(
    frames_root: str,
    fps: float,
    subset: str,
    clips_by_key: Dict[Tuple[str, str], List[List[int]]],
    frame_to_act: Dict[Tuple[str, str], Dict[int, str]],
    label_to_id: Dict[str, int],
    min_crop_ratio: float = 0.3,
    max_crop_ratio: float = 0.5,
) -> List[Path]:
    """
    Writes:
      - One JSON per (saved) clip: .../label/tal/<video>/<pid>/<video>_<pid>_<clipid>.json
      - One match.json per (video,pid): map of clip_key -> [start_abs0, end_abs0]
    Skips clips with no actions (after cropping/merging).
    """
    written: List[Path] = []

    frames_root = Path(frames_root)
    out_root = replace_frame_with_label_tal(frames_root)

    # Determine base frame (0-based) per video
    video_min_frame = defaultdict(lambda: math.inf)
    for (video, _pid), clips in clips_by_key.items():
        for clip in clips:
            if clip:
                video_min_frame[video] = min(video_min_frame[video], clip[0])
    for v in list(video_min_frame.keys()):
        if video_min_frame[v] is math.inf:
            video_min_frame[v] = 0

    matches_per_pid: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(dict)
    pid_dir_created: Dict[Tuple[str, str], bool] = defaultdict(bool)

    for (video, pid), clips in clips_by_key.items():
        f2a = frame_to_act.get((video, pid), {})

        for clipid, frames in enumerate(clips):
            if not frames:
                continue

            # random crop (if there is action; otherwise returns full)
            crop_s, crop_e = pick_random_crop_window(frames, f2a, min_crop_ratio, max_crop_ratio)
            frames = frames[crop_s:crop_e + 1]
            if not frames:
                continue

            # Build annotations by merging consecutive identical non-empty actions
            anns: List[Annotation] = []
            seg_start = None
            prev_act = None
            for i, f in enumerate(frames):
                act = f2a.get(f, "")
                if not act:
                    if prev_act is not None:
                        rel_start = seg_start
                        rel_end = i - 1
                        if rel_end >= rel_start:
                            start_s = round1(rel_start / fps)
                            end_s = round1((rel_end + 1) / fps)  # end is exclusive in seconds
                            anns.append(
                                Annotation(
                                    label=prev_act,
                                    segment=[start_s, end_s],
                                    segment_frames=[float(rel_start), float(rel_end + 1)],
                                    label_id=label_to_id[prev_act],
                                )
                            )
                        seg_start = None
                        prev_act = None
                else:
                    if prev_act is None:
                        seg_start = i
                        prev_act = act
                    elif act != prev_act:
                        rel_start = seg_start
                        rel_end = i - 1
                        start_s = round1(rel_start / fps)
                        end_s = round1((rel_end + 1) / fps)
                        anns.append(
                            Annotation(
                                label=prev_act,
                                segment=[start_s, end_s],
                                segment_frames=[float(rel_start), float(rel_end + 1)],
                                label_id=label_to_id[prev_act],
                            )
                        )
                        seg_start = i
                        prev_act = act
            # close any open segment at clip end
            if prev_act is not None and seg_start is not None:
                rel_start = seg_start
                rel_end = len(frames) - 1
                start_s = round1(rel_start / fps)
                end_s = round1((rel_end + 1) / fps)
                anns.append(
                    Annotation(
                        label=prev_act,
                        segment=[start_s, end_s],
                        segment_frames=[float(rel_start), float(rel_end + 1)],
                        label_id=label_to_id[prev_act],
                    )
                )

            # ===== Skip clips with no action =====
            if len(anns) == 0:
                continue

            # Prepare output paths (create pid dir lazily)
            if not pid_dir_created[(video, pid)]:
                pid_dir = (out_root / pid)
                pid_dir.mkdir(parents=True, exist_ok=True)
                pid_dir_created[(video, pid)] = True
            else:
                pid_dir = (out_root / pid)

            # Absolute indices and 0-based mapping for this (cropped) clip
            start_abs = frames[0]
            end_abs = frames[-1]
            base0 = int(video_min_frame[video])
            start_abs_0 = int(start_abs - base0)
            end_abs_0 = int(end_abs - base0)

            # Duration after crop
            n_frames = len(frames)
            duration_sec = round2(n_frames / fps)

            # Key & object
            clip_key = f"{video}_{pid}_{clipid}"
            tal_obj = {
                clip_key: {
                    "subset": subset,
                    "duration": duration_sec,
                    "fps": fps,
                    "annotations": [a.to_dict() for a in anns],
                }
            }

            out_path = pid_dir / f"{clip_key}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(tal_obj, f, ensure_ascii=False, indent=2)
            written.append(out_path)

            # Update matches for saved clip
            matches_per_pid[(video, pid)][clip_key] = [start_abs_0, end_abs_0]

        # Write match.json only if we saved any clips for this (video,pid)
        if matches_per_pid.get((video, pid)):
            pid_dir = (out_root / pid)
            pid_dir.mkdir(parents=True, exist_ok=True)
            match_path = pid_dir / "match.json"
            with open(match_path, "w", encoding="utf-8") as f:
                json.dump(matches_per_pid[(video, pid)], f, ensure_ascii=False, indent=2)
            written.append(match_path)


    # ------------------------------
    # Dataset statistics
    # ------------------------------
    total_clips = 0
    total_frames = 0
    total_action_frames = 0
    total_action_segments = 0

    for (video, pid), clips in clips_by_key.items():
        f2a = frame_to_act.get((video, pid), {})
        for frames in clips:
            if not frames:
                continue
            total_clips += 1
            total_frames += len(frames)

            # count frames that contain actions
            action_count = sum(1 for f in frames if f2a.get(f, ""))
            total_action_frames += action_count

            # number of distinct action segments in the clip
            segs = 0
            prev_act = None
            for f in frames:
                act = f2a.get(f, "")
                if act and act != prev_act:
                    segs += 1
                prev_act = act
            total_action_segments += segs

    if total_clips > 0:
        avg_clip_len = total_frames / total_clips
        avg_action_len = total_action_frames / total_action_segments if total_action_segments > 0 else 0
        avg_action_ratio = total_action_frames / total_frames
        print("\n===== Dataset Statistics =====")
        print(f"Total clips: {total_clips}")
        print(f"Average clip length (frames): {avg_clip_len:.2f}")
        print(f"Average action length (frames): {avg_action_len:.2f}")
        print(f"Average action ratio: {avg_action_ratio:.3f}")
        print("================================\n")

    return written

# ------------------------------
# Label map writer (optional)
# ------------------------------

def write_label_map(frames_root: str, label_to_id: Dict[str, int]):
    """
    Optionally write a global label map alongside the output root: .../label/tal/label_map.json
    """
    frames_root = Path(frames_root)
    out_root = replace_frame_with_label_tal(frames_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "label_map.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)
    return path

# ------------------------------
# Main
# ------------------------------

def main(csv_path: str, frames_root: str, fps: float = 60.0, subset: str = "Train", min_crop_ratio: float = 0.3, max_crop_ratio: float = 0.5):
    rows = load_and_process_csv(csv_path)
    label_to_id = LABEL_MAP
    clips_by_key = group_person_clips(rows)
    frame_to_act = build_frame_to_activity(rows)

    written = emit_clip_jsons_and_matches(
        frames_root=frames_root,
        fps=fps,
        subset=subset,
        clips_by_key=clips_by_key,
        frame_to_act=frame_to_act,
        label_to_id=label_to_id,
        min_crop_ratio=min_crop_ratio,
        max_crop_ratio=max_crop_ratio,
    )
    # Optionally write global label map
    # lm_path = write_label_map(frames_root, label_to_id)

    print(f"Done. Wrote {len(written)} files.")
    for p in written[:5]:
        print("  ", p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TAL labels from CSV and frame paths.")
    parser.add_argument("--root", type=str, required=True, help="Path to the root directory containing all video frames.")
    parser.add_argument("--fps", type=float, default=60.0, help="Video FPS (default: 60).")
    parser.add_argument("--subset", type=str, default="Train", help="Subset name (default: 'Train').")
    parser.add_argument("--min_ratio", type=float, default=0.3, help="Minimum crop ratio for action span (default: 0.3).")
    parser.add_argument("--max_ratio", type=float, default=0.5, help="Maximum crop ratio for action span (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cropping (default: 42).")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # === Traverse all subdirectories under frame_dir ===
    frame_root = Path(args.root)
    if not frame_root.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_root}")

    subdirs = [p for p in frame_root.iterdir() if p.is_dir()]
    print(f"Found {len(subdirs)} subdirectories in {frame_root}")

    for frame_path in subdirs:
        print(f"\nProcessing video frames in: {frame_path}")
        # Deduce CSV path (replace 'frame' with 'label/temporal' + ' exported.csv')
        csv_path = str(frame_path).replace("frame", "label/temporal").rstrip("/") + " exported.csv"

        try:
            main(
                csv_path=csv_path,
                frames_root=str(frame_path),
                fps=args.fps,
                subset=args.subset,
                min_crop_ratio=args.min_ratio,
                max_crop_ratio=args.max_ratio,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {frame_path}: {e}")
            continue
