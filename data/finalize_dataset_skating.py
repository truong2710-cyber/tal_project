#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple

# ------------------------------
# Feature collection
# ------------------------------

def collect_pose_features(root: Path, out_dir: Path) -> Set[str]:
    """
    Copy all *.npy from skating/features/<video>/<pid>/<clip>.npy
    into skating/final/pose_features/<clip>.npy

    Returns the set of clip basenames (without extension) that were copied.
    """
    src_root = root / "skating" / "features"
    copied_names: Set[str] = set()

    if not src_root.exists():
        print(f"[WARN] Features source not found: {src_root}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return copied_names

    out_dir.mkdir(parents=True, exist_ok=True)

    for video_dir in sorted(src_root.iterdir()):
        if not video_dir.is_dir():
            continue
        for pid_dir in sorted(video_dir.iterdir()):
            if not pid_dir.is_dir():
                continue
            for npy_path in sorted(pid_dir.glob("*.npy")):
                dest = out_dir / npy_path.name
                try:
                    shutil.copy2(npy_path, dest)
                    copied_names.add(npy_path.stem)  # basename without .npy
                except Exception as e:
                    print(f"[ERROR] Copy failed {npy_path} -> {dest}: {e}")
    return copied_names

# ------------------------------
# Train/Valid split (by parent video)
# ------------------------------

def pick_parent_split(
    tal_root: Path,
    valid_clip_names: Set[str],
    train_ratio: float,
    seed: int
) -> Tuple[Set[str], Set[str]]:
    """
    Build a deterministic Train/Valid split over parent videos under `tal_root`.
    Only consider parents that actually have at least one JSON with a matching .npy (by filename stem).
    Returns (train_parents, valid_parents) as sets of parent names (folder names).
    """
    parents_considered: List[str] = []

    # Scan parent dirs and include only those with at least one json whose stem ∈ valid_clip_names
    for video_dir in sorted(tal_root.iterdir()):
        if not video_dir.is_dir():
            continue
        has_match = False
        for pid_dir in sorted(video_dir.iterdir()):
            if not pid_dir.is_dir():
                continue
            for jp in pid_dir.glob("*.json"):
                if jp.name.lower() == "match.json":
                    continue
                if jp.stem in valid_clip_names:
                    has_match = True
                    break
            if has_match:
                break
        if has_match:
            parents_considered.append(video_dir.name)

    # Deterministic shuffle & split
    rng = random.Random(seed)
    rng.shuffle(parents_considered)

    n_train = int(round(len(parents_considered) * train_ratio))
    train_parents = set(parents_considered[:n_train])
    valid_parents = set(parents_considered[n_train:])

    print(f"Parents considered: {len(parents_considered)} "
          f"(Train={len(train_parents)}, Valid={len(valid_parents)})")
    return train_parents, valid_parents

# ------------------------------
# Merge TAL JSONs (filtered & subset reassigned)
# ------------------------------

def merge_tal_jsons_filtered(
    root: Path,
    out_json: Path,
    version: str,
    valid_clip_names: Set[str],
    train_parents: Set[str],
    valid_parents: Set[str],
) -> int:
    """
    Merge clip jsons from skating/label/tal/<parent>/<pid>/*.json (skip match.json),
    but ONLY include those whose filename (without .json) is present in valid_clip_names.

    Subset assignment:
      - If parent ∈ train_parents  -> "Train"
      - If parent ∈ valid_parents  -> "Validation"
      - Else (shouldn't happen)    -> "Train" by default

    Output format:
      {"version": version, "database": {clip_key: {...}, ...}}

    Returns number of clips merged.
    """
    tal_root = root / "skating" / "label" / "tal"
    if not tal_root.exists():
        print(f"[WARN] TAL root not found: {tal_root}")
        return 0

    db: Dict[str, Any] = {}
    merged = 0

    for parent_dir in sorted(tal_root.iterdir()):
        if not parent_dir.is_dir():
            continue
        parent = parent_dir.name

        # Determine subset for this parent
        if parent in train_parents:
            subset_value = "Train"
        elif parent in valid_parents:
            subset_value = "Validation"
        else:
            subset_value = "Train"  # default fallback

        for pid_dir in sorted(parent_dir.iterdir()):
            if not pid_dir.is_dir():
                continue
            for jp in sorted(pid_dir.glob("*.json")):
                if jp.name.lower() == "match.json":
                    continue
                clip_stem = jp.stem  # e.g. video_pid_clipid
                if clip_stem not in valid_clip_names:
                    continue  # ignore json without corresponding npy

                try:
                    with jp.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[ERROR] Failed reading {jp}: {e}")
                    continue

                if not isinstance(data, dict) or len(data) == 0:
                    continue

                # Merge keys (usually just one)
                for k, v in data.items():
                    # ensure we only take wanted key
                    if k != clip_stem and k not in valid_clip_names:
                        continue

                    # Reassign subset in the clip payload
                    # (be defensive about key casing)
                    if isinstance(v, dict):
                        if "subset" in v:
                            v["subset"] = subset_value

                    if k in db:
                        print(f"[WARN] duplicate key '{k}' in {jp}, overwriting.")
                    db[k] = v
                    merged += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    final_obj = {"version": version, "database": db}
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)

    return merged

# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect pose features, split parents into Train/Validation, and merge TAL JSONs (filtered to matching clips)."
    )
    parser.add_argument("--root", type=str, default=".", help="Project root containing 'skating/'.")
    parser.add_argument("--version", type=str, default="Skating-60fps", help="Version string for the merged annotations.")
    parser.add_argument("--features_out_dir", type=str, default="skating/final/pose_features",
                        help="Destination folder to collect all .npy features.")
    parser.add_argument("--annotations_out_dir", type=str, default="skating/final/annotations",
                        help="Destination folder for the merged annotations JSON.")
    parser.add_argument("--annotations_filename", type=str, default="skating.json",
                        help="Filename for the merged annotations JSON.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of parent videos to use for Train (rest go to Validation).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the parent split.")
    args = parser.parse_args()

    root = Path(args.root)
    features_out = root / args.features_out_dir
    ann_out_dir = root / args.annotations_out_dir
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    ann_out_path = ann_out_dir / args.annotations_filename

    print("== Collecting pose features ==")
    valid_clip_names = collect_pose_features(root, features_out)
    print(f"Copied {len(valid_clip_names)} feature files to: {features_out}")

    tal_root = root / "skating" / "label" / "tal"
    if not tal_root.exists():
        raise FileNotFoundError(f"TAL root not found: {tal_root}")

    print("\n== Splitting parent videos into Train/Validation ==")
    train_parents, valid_parents = pick_parent_split(
        tal_root=tal_root,
        valid_clip_names=valid_clip_names,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    print("\n== Merging TAL annotations (filtered + subset reassigned) ==")
    n_json = merge_tal_jsons_filtered(
        root=root,
        out_json=ann_out_path,
        version=args.version,
        valid_clip_names=valid_clip_names,
        train_parents=train_parents,
        valid_parents=valid_parents,
    )
    print(f"Merged {n_json} clip entries into: {ann_out_path}")
    if n_json != len(valid_clip_names):
        print(f"[INFO] Filter applied: {len(valid_clip_names)} features vs {n_json} json entries merged.")

    print("\n== Split summary ==")
    print(f"Train parents: {len(train_parents)}  -> {sorted(list(train_parents))[:5]}{' ...' if len(train_parents)>5 else ''}")
    print(f"Valid parents: {len(valid_parents)}  -> {sorted(list(valid_parents))[:5]}{' ...' if len(valid_parents)>5 else ''}")

if __name__ == "__main__":
    main()
