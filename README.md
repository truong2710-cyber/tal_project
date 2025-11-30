# TAL Project

This repository implements a **Temporal Action Localization (TAL)** pipeline that combines [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [OpenMMLab](https://github.com/open-mmlab/mmpose) for **action recognition and localization** on the *shot* and *skating* datasets using feautures from poses.

---

## 📦 1. Clone the Repository

```bash
git clone https://github.com/truong2710-cyber/tal_project.git
cd tal_project
```

---

## ⚙️ 2. Environment Setup

Run

```bash
bash install.sh
```

to install. This will create 2 conda environments: ``actionformer`` and ``openmmlab``.

---

## 🧠 3. Training & Evaluation Pipeline

### 3.1. Raw Data Preparation

Extract raw data and place them under the `data/` directory as follows:

#### Shot Dataset
```
data/shot/
├── raw/
├── label/
│   ├── temporal/
│   └── tracking/
```

#### Skating Dataset
```
data/skating/
├── frame/
├── label/
│   ├── temporal/
│   └── tracking/
```

---

### 3.2. Keypoint Extraction

1. Decode the shot videos:
   ```bash
   python decode_videos.py
   ```
   Frames will be saved to `data/shot/frame/`.

2. Download the pretrained model [epoch_240.pth](https://huggingface.co/truongvu2710/keypoints/resolve/main/pretrained_weights/epoch_240.pth?download=true) and place it in:
   ```
   mmpose/pretrained_weights/
   ```

3. Extract keypoints using distributed GPU processing:
   ```bash
   cd ../mmpose
   bash extract_keypoints_frame_dist.sh ../data/shot/frame
   bash extract_keypoints_frame_dist.sh ../data/skating/frame
   ```

After completion, keypoints will be stored in:
```
data/<dataset_name>/keypoints/
```

**Optional:** You can download the pre-extracted keypoints at https://huggingface.co/truongvu2710/keypoints/tree/main.

---

### 3.3. TAL Label Parsing

We use the **THUMOS14** label format as implemented in `actionformer_release/` and generate TAL labels from temporal and tracking annotations.

```bash
cd data

# For shot dataset
python parse_tal_labels_shot.py --root .

# For skating dataset
python parse_tal_labels_skating.py --root skating/keypoints/
```

---

### 3.4. Feature Generation

This step extracts clips from consecutive frames and creates ActionFormer input features.

By default:
- 16 frames per feature  
- Frame stride = 8
- IoU threshold = 0.3 (for matching tracked boxes with keypoints)

```bash
cd ../data
python prepare_features_shot.py --root . --K [num_frames_per_feat] --S [frame_stride] --iou_thr [iou_thresh]
python prepare_features_skating.py --root . --K [num_frames_per_feat] --S [frame_stride] --iou_thr [iou_thresh]
```

Output directory:
```
data/<dataset_name>/features/
```

---

### 3.5. Dataset Finalization

Aggregate features and labels to form the final dataset in THUMOS14 format for ActionFormer training and evaluation.

```bash
python finalize_dataset_shot.py --root .
python finalize_dataset_skating.py --root .
```

Move results to ActionFormer’s data directory:
```bash
mkdir -p ../actionformer_release/data/shot
mkdir -p ../actionformer_release/data/skating

mv shot/final/* ../actionformer_release/data/shot
mv skating/final/* ../actionformer_release/data/skating
```

---

### 3.6. Training

Train the ActionFormer model using the generated datasets:

```bash
cd ../actionformer_release
python train.py ./configs/shot_pose.yaml --output test
python train.py ./configs/skating_pose.yaml --output test
```

---

### 3.7. Evaluation

Evaluate trained models:

```bash
python eval.py ./configs/shot_pose.yaml ./ckpt/shot_pose_test
python eval.py ./configs/skating_pose.yaml ./ckpt/skating_pose_test
```

---

## 🎬 4. Demo

To run inference and visualize results:

```bash
bash inference.sh <INPUT_PATH> <af_config_yaml> <af_ckpt> [output_dir]
```

- `INPUT_PATH` can be:
  - a directory containing frames, or  
  - a path to an `.mp4` video.

For example:
```
bash inference.sh '../data/shot/raw/2025-01-31 Nashville at Buffalo/ShotEvent/2025-01-31_Nashville_at_Buffalo_ShotEvent_1_023_ev_f8o9gx3_snap_89_29.mp4' configs/shot_pose.yaml ckptshot_pose_test/
```

The output video visualization will be saved at:
```
output_dir/inference.mp4
```

---

## 📚 Citation

If you find this project helpful in your research, please cite the original repositories:

- **[ActionFormer](https://github.com/happyharrycn/actionformer_release)**
- **[OpenMMLab (MMPose)](https://github.com/open-mmlab/mmpose)**

---

## 🧾 License

This project follows the licenses of the original repositories it builds upon.





