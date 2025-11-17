# python imports
import argparse
import csv
import os
import glob
import time
from pprint import pprint
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed
from libs.utils.train_utils import AverageMeter

# add path for importing TAL label maps
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from data.parse_tal_labels_skating import LABEL_MAP as SKATING_LABEL_MAP
from data.parse_tal_labels_shot import LABEL_MAP as SHOT_LABEL_MAP


class SimplePoseEvalDataset(Dataset):
    """
    Minimal evaluation Dataset.

    Folder layout (example):
      feat_folder/
        12/
          12_0.npy
          12_1.npy
          match.json  # {"12_0":[start,end], "12_1":[start,end], ...} (inclusive, 0-based)
        27/
          27_0.npy
          match.json
        ...

    Each *.npy is a sample. Duration (in seconds) is computed from match.json and default_fps:
        duration = (end - start + 1) / default_fps
    """

    def __init__(
        self,
        feat_folder: str,   # root folder containing pid subfolders
        feat_stride: int,   # frames per feature step (used downstream)
        num_frames: int,    # temporal window length used when creating feats (token receptive field)
        default_fps: float, # fps used to convert [start,end] frames -> seconds
        input_dim: int,     # expected feature dimension (D)
    ):
        assert os.path.isdir(feat_folder), f"feat_folder not found: {feat_folder}"
        self.feat_folder = feat_folder
        self.feat_stride = int(feat_stride)
        self.num_frames = int(num_frames)
        self.default_fps = float(default_fps)
        self.input_dim = int(input_dim)

        # Build an index of samples: id, paths, fps, duration, and frame span
        self.data_list: List[Dict[str, Any]] = self._scan_folder()

    def _scan_folder(self) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []
        root = Path(self.feat_folder)

        for pid_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            match_path = pid_dir / "match.json"
            if not match_path.exists():
                continue
            try:
                with match_path.open("r", encoding="utf-8") as f:
                    match_map = json.load(f)  # {clip_stem: [start, end], ...}
            except Exception:
                continue

            for npy_path in sorted(pid_dir.glob("*.npy")):
                clip_stem = npy_path.stem  # e.g., "12_0"
                if clip_stem not in match_map:
                    continue

                start_f, end_f = match_map[clip_stem]  # inclusive, 0-based (parent video frames)
                start_f = int(start_f)
                end_f = int(end_f)
                duration_sec = (end_f - start_f + 1) / self.default_fps

                # Light sanity check
                try:
                    arr = np.load(npy_path, mmap_mode="r")
                    if arr.ndim != 2:
                        continue
                    T, D = arr.shape
                    # Optionally enforce D == input_dim
                    # if D != self.input_dim: continue
                except Exception:
                    continue

                data_list.append({
                    "id": clip_stem,                             # sample id = file stem
                    "feat_path": str(npy_path),                  # path to .npy
                    "fps": self.default_fps,                     # single fps used for all
                    "duration": round(float(duration_sec), 2),   # seconds
                    "frame_start": start_f,                      # absolute 0-based parent-video frame index (inclusive)
                    "frame_end": end_f,                          # absolute 0-based parent-video frame index (inclusive)
                })

        return data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
          {
            'video_id'       : <clip_stem>,
            'feats'          : torch.FloatTensor [C x T], where C = input_dim,
            'fps'            : float,
            'duration'       : float (seconds),
            'feat_stride'    : int (frames/feature),
            'feat_num_frames': int (token receptive field in frames),
            'frame_start'    : int,  # absolute 0-based parent-video frame (inclusive)
            'frame_end'      : int,  # absolute 0-based parent-video frame (inclusive)
          }
        """
        item = self.data_list[idx]

        feats = np.load(item["feat_path"]).astype(np.float32)  # (T, D)
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose(1, 0)))  # (D, T)

        data_dict = {
            "video_id": item["id"],
            "feats": feats,                              # C x T
            "fps": item["fps"],
            "duration": item["duration"],
            "feat_stride": self.feat_stride,
            "feat_num_frames": self.num_frames,
            "frame_start": item["frame_start"],
            "frame_end": item["frame_end"],
        }
        return data_dict


def visualize_actions_to_video(
    frames_dir: str,
    tracking_csv: str,
    detections: List[Dict[str, Any]],
    out_mp4: str,
    fps: float = 60.0,
    fourcc: str = "mp4v",  # or "avc1" if your ffmpeg/OpenCV supports H.264
    box_color: Tuple[int, int, int] = (0, 0, 255),  # BGR (red)
    box_thickness: int = 3,
    font_scale: float = 0.8,
    font_thickness: int = 2,
) -> None:
    """
    Create a visualization MP4 from frames + tracking + action detections.

    Args:
        frames_dir: folder containing the frames (filenames should match the 'Name' column in CSV).
        tracking_csv: CSV with columns frame_number,Name,id,x1,y1,x2,y2
        detections: list of dicts with keys:
           - 'pid' (int)
           - 't_start_frame_abs' (int, absolute parent-video frame index, inclusive)
           - 't_end_frame_abs'   (int, absolute parent-video frame index, inclusive)
           - 'label' (str)
           - 'score' (float)
           - (t_start_abs, t_end_abs may also be present but not required here)
        out_mp4: output video path.
        fps: frames per second for output video.
        fourcc: OpenCV fourcc code (default mp4v).
        box_color: BGR color for the acting bbox.
        box_thickness: rectangle thickness.
        font_scale: text scale.
        font_thickness: text thickness.
    """
    frames_dir = Path(frames_dir)

    # --- 1) Load tracking into: frame_number -> {pid: (x1,y1,x2,y2)} and also frame_number -> representative frame filename
    frame_to_pid_box: DefaultDict[int, Dict[int, Tuple[int, int, int, int]]] = defaultdict(dict)
    frame_to_name: Dict[int, str] = {}

    with open(tracking_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        req = ["frame_number", "Name", "id", "x1", "y1", "x2", "y2"]
        if any(c not in reader.fieldnames for c in req):
            raise ValueError(f"tracking_csv must contain columns: {req}. Got: {reader.fieldnames}")

        for row in reader:
            fnum = int(row["frame_number"])
            name = row["Name"].strip()
            pid = int(row["id"])
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            frame_to_pid_box[fnum][pid] = (x1, y1, x2, y2)
            # store one file name per frame_number (assumes consistent naming)
            if fnum not in frame_to_name:
                frame_to_name[fnum] = name

    if not frame_to_name:
        raise RuntimeError("No frames found from tracking CSV.")

    # --- 2) Build per-frame action map:
    # For each frame, for each pid acting, choose the *best* (highest score) action spanning that frame.
    # We’ll store: frame_actions[fnum][pid] = (label, score)
    frame_actions: DefaultDict[int, Dict[int, Tuple[str, float]]] = defaultdict(dict)

    for det in detections:
        pid = int(det["pid"])
        fs = int(det["t_start_frame_abs"])  # inclusive
        fe = int(det["t_end_frame_abs"])    # inclusive
        label = str(det.get("label", "Action"))
        score = float(det.get("score", 0.0))

        if fe < fs:
            continue

        for f in range(fs, fe + 1):
            prev = frame_actions[f].get(pid, None)
            if (prev is None) or (score > prev[1]):  # keep highest score per (frame, pid)
                frame_actions[f][pid] = (label, score)

    # --- 3) Determine frame range to render (min..max present in names)
    all_frames_sorted = sorted(frame_to_name.keys())
    first_frame = all_frames_sorted[0]
    last_frame = all_frames_sorted[-1]

    # --- 4) Initialize video writer using the first available frame to get resolution
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    # Find the first existing frame image on disk
    first_img = None
    for fnum in all_frames_sorted:
        img_path = frames_dir / frame_to_name[fnum]
        if img_path.exists():
            first_img = cv2.imread(str(img_path))
            if first_img is not None:
                break
    if first_img is None:
        raise RuntimeError("Could not read any frame image from frames_dir.")

    H, W = first_img.shape[:2]
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (W, H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_mp4}")

    # --- 5) Render loop
    font = cv2.FONT_HERSHEY_SIMPLEX
    for fnum in range(first_frame, last_frame + 1):
        name = frame_to_name.get(fnum, None)
        if name is None:
            # frame index exists in CSV mapping? if not, we still attempt to write a blank frame of same size
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            img_path = frames_dir / name
            frame = cv2.imread(str(img_path))
            if frame is None:
                # missing image → draw black frame so video is continuous
                frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Which pids act on this frame?
        acting = frame_actions.get(fnum, {})

        # Draw only acting pid boxes (highlight)
        if acting:
            pid_boxes = frame_to_pid_box.get(fnum, {})
            y_text = 30
            for pid, (label, score) in acting.items():
                if pid in pid_boxes:
                    x1, y1, x2, y2 = pid_boxes[pid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
                    text = f"pid {pid}: {label} ({score:.2f})"
                    # put text (shadow for readability)
                    cv2.putText(frame, text, (x1, max(20, y1 - 10)), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(frame, text, (x1, max(20, y1 - 10)), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                else:
                    # No bbox available for this pid in this frame; still show a banner
                    text = f"pid {pid}: {label} ({score:.2f})"
                    cv2.putText(frame, text, (10, y_text), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(frame, text, (10, y_text), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    y_text += int(30 * font_scale + 10)

        writer.write(frame)

    writer.release()
    print(f"[OK] Wrote visualization to: {out_mp4}")

################################################################################
def main(args):
    if args.out_video:
        assert os.path.exists(args.frame_dir), "If --out_video is specified, --frame_dir must also exist."
        assert os.path.exists(args.csv_file), "If --out_video is specified, --csv_file must also exist."
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    if cfg['dataset_name'] == 'skating':
        rev_label_map = {v: k for k, v in SKATING_LABEL_MAP.items()}
    elif cfg['dataset_name'] == 'shot':
        rev_label_map = {v: k for k, v in SHOT_LABEL_MAP.items()}

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    # val_dataset = make_dataset(
    #     cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    # )
    val_dataset = SimplePoseEvalDataset(
        feat_folder=args.feat_folder,
        feat_stride=cfg['dataset']['feat_stride'],
        num_frames=cfg['dataset']['num_frames'],
        default_fps=args.fps,
        input_dim=cfg['dataset']['input_dim'],
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()

    # set up meters
    batch_time = AverageMeter()
    print_freq = args.print_freq
    # switch to evaluate mode
    model.eval()
    
    results = []  # list of dicts 

    # loop over validation set
    start = time.time()

    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)  # list of dicts per clip (batch)

            # Each entry in output corresponds to one sample in video_list
            for sample, out in zip(video_list, output):
                segs = out["segments"]   # (N, 2)
                labels = out["labels"]
                scores = out["scores"]
                vid = out["video_id"]

                if segs is None or segs.shape[0] == 0:
                    continue

                # Retrieve frame offset and fps from the dataset batch
                frame_start = sample["frame_start"]
                fps = sample["fps"]

                # convert from local (clip) seconds → absolute video seconds
                clip_offset_sec = frame_start / fps

                for i in range(segs.shape[0]):
                    if scores[i] < args.score_thresh:
                        continue

                    t_start_abs = float(segs[i, 0].item()) + clip_offset_sec
                    t_end_abs = float(segs[i, 1].item()) + clip_offset_sec

                    results.append({
                        "pid": vid.rsplit('_', 1)[0],
                        "t_start_abs": t_start_abs,
                        "t_end_abs": t_end_abs,
                        "t_start_frame_abs": int(round(t_start_abs * fps)),
                        "t_end_frame_abs": int(round(t_end_abs * fps)),
                        "label": rev_label_map[int(labels[i].item()) if torch.is_tensor(labels[i]) else int(labels[i])],
                        "score": float(scores[i].item()) if torch.is_tensor(scores[i]) else float(scores[i]),
                    })


        # printing
        if (iter_idx != 0) and (iter_idx % print_freq == 0):
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            print("Test: [{0:05d}/{1:05d}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})".format(
                    iter_idx, len(val_loader), batch_time=batch_time))

    # breakpoint()
    end = time.time()
    print("Total time for detection: {:0.2f} sec".format(end - start))

    if args.out_video:
        start = time.time()
        visualize_actions_to_video(
            frames_dir=args.frame_dir,
            tracking_csv=args.csv_file,  
            detections=results,
            out_mp4=args.out_video,
            fps=args.fps,
        )
        end = time.time()
        print("Total time for visualization: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('feat_folder', type=str, metavar='DIR',
                        help='path to the feature folder for evaluation')
    parser.add_argument('--out_video', type=str,
                        help='path to the output visualization video')
    parser.add_argument('--frame_dir', type=str, default='',
                        help='path to the frame folder for visualization')
    parser.add_argument('--csv_file', type=str, default='',
                        help='path to the tracking csv folder for visualization')
    parser.add_argument('-fps', type=int, default=60,
                        help='default fps for the evaluation dataset')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-s', '--score-thresh', default=0.5, type=float,
                        help='score threshold to filter outputs (default: 0.1)')
    args = parser.parse_args()
    main(args)
