#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm


def list_images_sorted(frame_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in frame_dir.iterdir() if p.suffix.lower() in exts]
    # natural sort by numeric chunk if present (e.g., ..._000001.jpg)
    def key(p: Path):
        s = p.stem
        num = ""
        for ch in reversed(s):
            if ch.isdigit():
                num = ch + num
            else:
                break
        return (s[: len(s) - len(num)], int(num) if num else -1, s)
    files.sort(key=key)
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Track people across a folder of frames and export CSV (Name,id,x1,y1,x2,y2)."
    )
    ap.add_argument("--frame_dir", required=True, type=str, help="Directory of input frames.")
    ap.add_argument("--frame_rate", type=int, default=60, help="Frame rate of the input video.")
    ap.add_argument("--out_csv", required=True, type=str, help="Path to output CSV.")
    ap.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt).",
    )
    ap.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu (auto if None).")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    # ByteTrack knobs (safe defaults)
    ap.add_argument("--bt_track_thresh", type=float, default=0.5, help="ByteTrack track threshold.")
    ap.add_argument("--bt_track_buffer", type=int, default=30, help="Frames to keep lost tracks.")
    ap.add_argument("--bt_match_thresh", type=float, default=0.8, help="Matching threshold.")
    ap.add_argument(
        "--person_class_id",
        type=int,
        default=0,
        help="COCO person class id (YOLOv8 COCO uses 0 for person).",
    )
    args = ap.parse_args()

    frame_dir = Path(args.frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frame_dir}")

    image_paths = list_images_sorted(frame_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {frame_dir}")

    # Load detector
    model = YOLO(args.model)
    # Load tracker
    tracker = sv.ByteTrack(
        args.bt_track_thresh,
        args.bt_track_buffer,
        args.bt_match_thresh,
        args.frame_rate,
    )

    # Prepare CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "Name", "id", "x1", "y1", "x2", "y2"])

        for frame_number, img_path in enumerate(tqdm(image_paths, desc="Tracking people")):
            # Read image (BGR)
            im = cv2.imread(str(img_path))
            if im is None:
                continue

            # Run YOLOv8 on this frame (single-image)
            results = model.predict(
                im,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )

            if not results:
                continue

            r = results[0]
            # Convert to Supervision Detections
            det = sv.Detections.from_ultralytics(r)

            # Filter only "person" class
            if det.class_id is not None:
                person_mask = det.class_id == args.person_class_id
                det = det[person_mask]

            # Update tracker with detections
            tracks = tracker.update_with_detections(det)

            # tracks is also a Detections object with .tracker_id
            if tracks.tracker_id is None:
                continue

            # For each track, write row
            for bbox_xyxy, tid in zip(tracks.xyxy, tracks.tracker_id):
                x1, y1, x2, y2 = bbox_xyxy.tolist()
                # round to ints for CSV
                x1, y1, x2, y2 = map(lambda v: int(round(float(v))), (x1, y1, x2, y2))
                writer.writerow([frame_number, img_path.name, int(tid), x1, y1, x2, y2])

    print(f"Done. Wrote Tracking CSV to: {out_path}")


if __name__ == "__main__":
    main()
