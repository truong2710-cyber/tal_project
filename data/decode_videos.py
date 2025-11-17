#!/usr/bin/env python3
import os
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count

def decode_one(video_path: Path, raw_root: Path, frame_root: Path):
    # replicate directory structure from raw/ -> frame/
    rel_path = video_path.relative_to(raw_root)
    out_dir = frame_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_name = video_path.stem  # e.g., clip_001
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ {video_path} → {out_dir} ({total} frames)")

    clip_dir = out_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = clip_dir / f"{clip_name}_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_id += 1

    # ✅ delete original video if decoding succeeded
    if frame_id > 1:
        try:
            # video_path.unlink()  # deletes the file
            print(f"🗑️  Deleted {video_path} after decoding ({frame_id} frames)")
        except Exception as e:
            print(f"⚠️ Could not delete {video_path}: {e}")
    else:
        print(f"⚠️ Skipped deletion: no frames decoded from {video_path}")

    cap.release()
    print(f"✅ Done {video_path} → {frame_id - 1} frames")

def find_videos(raw_root: Path):
    # find all videos under ShotEvent directories
    return list(raw_root.glob("*/ShotEvent/*.mp4"))

def main():
    raw_root = Path("shot/raw").resolve()
    frame_root = Path("shot/frame").resolve()
    frame_root.mkdir(parents=True, exist_ok=True)

    videos = find_videos(raw_root)
    if not videos:
        print(f"No videos found in {raw_root}")
        return

    workers = max(cpu_count() // 2, 1)
    print(f"Found {len(videos)} videos under {raw_root}")
    print(f"Decoding with {workers} workers...")

    with Pool(workers) as pool:
        pool.starmap(decode_one, [(v, raw_root, frame_root) for v in videos])

if __name__ == "__main__":
    main()
