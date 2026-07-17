# Usage Guide

This guide covers the project background, the full repository layout, the dataset, environment setup, and how to run every script — including the ones used to reproduce the validation results reported in the paper.

If you just want to try inference with the included trained model, see the Quick start in [README.md](README.md).

## 1. Background

Structural translational displacement is a key indicator of structural damage and dynamic response, but contact sensors (e.g. LVDTs) are often impractical in destructive or field-monitoring tests. This project provides a vision-based alternative: a chessboard target is mounted on the structure, tracked in video, and its motion is converted into displacement.

The framework has two stages:

1. **Improved YOLO detector (Y-BWA-W)** — a YOLOv5 detector with a proposed **Black-White Attention (BWA)** module (a Contrast-Enhancement Block `CEB` + Corner-Attention Block `CAB`, see `nets/CSPdarknet.py`) inserted into the CSPDarknet backbone to better localize chessboard targets, plus **Weighted Non-Maximum Suppression (WNMS)** (`utils/utils_bbox.py`, `non_max_suppression(..., wnms=True)`) to stabilize bounding boxes across frames.
2. **Deep SORT tracker** (`nets/deep_sort.py`, `nets/sort/`) — combines Kalman-filter motion prediction with a learned appearance descriptor to keep the target's identity through short-term occlusion.

In-plane (horizontal) displacement is read directly from the tracked bounding-box position; out-of-plane (depth) displacement is estimated from the change in the target's apparent size using the similar-triangles relationship in `depth_measurement.py`.

Validation in the paper uses three experiments, all reproducible from this repo:

- an outdoor **shaking-table test** (dynamic, in-plane) — `validation.py: validate_EQ1()`
- a **static cyclic RC shear-wall test** (in-plane) — `validation.py: validate_PCW2()`
- a **small-vibration-platform depth test** with a frontal and an angled camera (out-of-plane) — `depth_measurement.py`

## 2. Repository structure

```
validation/
├── main.py                  # Real-time YOLO + Deep SORT demo (webcam or video file, CLI)
├── predict.py                # Single-image / batch / FPS / heatmap / ONNX-export modes
├── predict_yolo.py           # Earlier/alternate variant of predict.py
├── validation.py              # Validator class + in-plane validation experiments (Section 5.1)
├── depth_measurement.py       # Depth_Estimator class + out-of-plane validation experiment (Section 5.2)
├── target.py                  # Standalone perspective-correction routine for the angled camera
├── plot_template.py           # Shared matplotlib helper for paper-style figures
├── test.py, util.py           # Small one-off helper scripts (not part of the main pipeline)
├── yolo.py                     # YOLO wrapper class: model loading, inference, heatmap, ONNX export
│
├── nets/
│   ├── CSPdarknet.py           # Backbone + BWA attention modules (CEB, CAB, plus SE/ECA/CBAM/CA baselines)
│   ├── yolo.py, yolo_training.py  # YOLO head/body and loss
│   ├── ConvNext.py, Swin_transformer.py  # Alternative backbones supported by yolo.py's `backbone` option
│   ├── deep_sort.py            # DeepSort class (tracker + appearance extractor)
│   ├── sort/                    # Kalman filter, Hungarian matching, IoU/appearance metrics
│   └── deep/                    # Appearance ReID network (`model.py`) and its trainer (`train_deepsort.py`)
│
├── utils/
│   ├── utils.py, dataloader.py, augmentation.py   # Config, data loading, augmentation
│   ├── utils_bbox.py            # Box decoding + (weighted) non-max suppression
│   ├── utils_fit.py, utils_map.py  # Training loop / mAP evaluation helpers
│   └── correct_perspective.py   # Chessboard-corner detection + perspective rectification
│
├── utils_ds/                    # Deep SORT support: drawing, logging, YAML config parsing
├── utils_coco/                  # COCO-format mAP evaluation helpers
│
├── model_data/
│   ├── best_epoch_weights.pth   # Trained Y-BWA-W YOLO detector weights
│   ├── ckpt_target_epoch50.t7   # Trained Deep SORT appearance (ReID) weights
│   ├── yolov5_s.pth             # ImageNet-pretrained backbone (used for training only)
│   ├── target_classes.txt, coco_classes.txt, yolo_anchors.txt  # Class/anchor definitions
│   ├── deep_sort.yaml           # Deep SORT hyperparameters (used by main.py)
│   └── 2007_train.txt / 2007_val.txt / 2007_test.txt  # VOC-style annotation index files (see Dataset section)
│
├── experimental_data/            # Raw recordings + sensor logs (not tracked in git — see Dataset section)
│   ├── front_camera/, side_camera/   # Video recordings (dynamic, static, depth tests)
│   └── raw_data/                     # Sensor logs (.bsi / .txt)
│
├── logs/                          # Script outputs: displacement curves, detected positions, etc. (not tracked in git)
└── detections/                    # Script outputs: cropped/annotated detection images (not tracked in git)
```

`experimental_data/`, `logs/`, and `detections/` are excluded from version control via `.gitignore` (see the Dataset section below) — they're regenerated by running the scripts, or restored from the Zenodo archive.

## 3. Dataset

The Y-BWA-W detector was trained on a purpose-built database of chessboard-target images, described in Section 3 of the paper:

- **2,132 annotated high-resolution images**: 800 frames from dynamic (shaking-table) tests and 1,332 frames from static tests, ranging from 1920×1080 up to 6000×4000 pixels, covering varying target scale/orientation/position plus challenging conditions (illumination changes, partial occlusion, motion blur, complex backgrounds).
- **Labeling process**: initial bounding boxes were generated automatically via template matching (fast normalized cross-correlation), then manually checked and refined.
- **Augmentation**: vertical/horizontal flip, random 90° rotation, translation/scale/rotation, perspective transform, hue-saturation-lightness and brightness/contrast jitter, plus synthetic shadows, sun flares, Gaussian/ISO noise, and Gaussian blur, applied in randomized combinations. Both original and augmented images were then cropped to a standardized 1920×1080 resolution, giving **10,593 cropped images** in total.
- **Train / validation split**: 1,492 images (70%, augmented) for training, 640 images (30%, held out and *not* augmented) for validation.
- **Annotation format**: the same VOC-style convention used by [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch) — one line per image, `image_path xmin,ymin,xmax,ymax,class_id ...`. In this repo, the index files live in `model_data/2007_train.txt`, `2007_val.txt`, and `2007_test.txt`, with the class list in `model_data/target_classes.txt` and anchor boxes in `model_data/yolo_anchors.txt`.

**Availability**: this repository ships the trained weights (`model_data/best_epoch_weights.pth`, `model_data/ckpt_target_epoch50.t7`) and the annotation index files, so no dataset download is needed to run inference or the validation scripts as-is.

The raw labelled images (the 2,132-image database above) and the raw experimental recordings/sensor logs (`experimental_data/`) are too large for this git repository and are instead openly published on Zenodo under **CC-BY 4.0**:

> Zhang, H., Cheng, X., Li, Y., Guan, H. (2026). *Vision-based Real-time Structural Displacement Measurement Using Deep Learning — Dataset* [Data set]. Zenodo. [doi.org/10.5281/zenodo.21400638](https://doi.org/10.5281/zenodo.21400638)
>
> Concept DOI (always resolves to the latest version): [doi.org/10.5281/zenodo.21400637](https://doi.org/10.5281/zenodo.21400637)

Two archives are provided:

| Archive | Size | Contents | Extract to |
|---|---|---|---|
| `VOCdevkit.zip` | 24.2 GB | The full 2,132-image labelled database (train + validation), in the VOC-style layout expected by `model_data/2007_train.txt` / `2007_val.txt` / `2007_test.txt` | Repository root, so `VOCdevkit/VOC2007/JPEGImages/...` sits alongside `model_data/`, `nets/`, etc. |
| `experimental_data.zip` | 19.6 GB | Raw camera recordings + sensor logs for the three validation experiments | Repository root (creates/replaces `experimental_data/`) |

```bash
# from the repository root
curl -L -o VOCdevkit.zip "https://zenodo.org/records/21400638/files/VOCdevkit.zip?download=1"
curl -L -o experimental_data.zip "https://zenodo.org/records/21400638/files/experimental_data.zip?download=1"

unzip VOCdevkit.zip -d .
unzip experimental_data.zip -d .
```

## 4. Environment setup

The scripts were developed and run in a conda environment named `yolo` (Python 3.8, PyTorch 2.4.1 + CUDA 11.8):

```bash
conda create -n yolo python=3.8
conda activate yolo
pip install -r requirements.txt
```

`requirements.txt` pins CPU/CUDA-agnostic packages; PyTorch itself is listed as `torch==2.4.1+cu118` for a CUDA 11.8 GPU. If you don't have a matching GPU/driver, install a CPU or different-CUDA build from [pytorch.org](https://pytorch.org) instead, and set `"cuda": False` in `yolo.py`'s `_defaults` (or pass `--device cpu` to `main.py`).

All commands below assume the environment is active and the current directory is `validation/`:

```bash
conda activate yolo
```

## 5. Running the scripts

### 5.1 `predict.py` — single image / batch / webcam / FPS / heatmap / ONNX

The single entry point for general inference. Open the file and set the `mode` variable near the bottom (each mode's own parameters live directly below it):

| `mode` | What it does |
|---|---|
| `"predict"` | Prompts for an image path, shows the detection result |
| `"video"` | Runs detection on a webcam or video file |
| `"fps"` | Benchmarks inference speed on a single image |
| `"dir_predict"` | Batch-detects every image in `dir_origin_path`, saves annotated copies to `dir_save_path` |
| `"heatmap"` | Saves an Eigen-CAM attention heatmap for an input image (Fig. 9 in the paper) |
| `"export_onnx"` | Exports the model to ONNX |

```bash
python predict.py
```

### 5.2 `main.py` — real-time detection + Deep SORT tracking

A CLI demo that runs the full Y-BWA-W + Deep SORT pipeline on a video file or webcam and can save an annotated video and per-frame bounding-box logs.

```bash
python main.py --input_path path/to/video.mp4 --save_path video/ --save_txt logs/deepSORT/
# or, for a live webcam:
python main.py --camera 0
```

Key options: `--device` (`cuda`/`cpu`), `--display` (show a live window), `--config_deepsort` (points to `model_data/deep_sort.yaml`, which sets Deep SORT's distance thresholds, max age, etc.).

### 5.3 `validation.py` — in-plane validation (Section 5.1)

The `Validator` class handles frame sorting, position→center-of-box conversion, outlier removal, and RMSE calculation against sensor-measured displacement. Two experiment-specific functions reproduce the paper's figures:

- `validate_EQ1()` — dynamic shaking-table test vs. LVDT (Fig. 14), RMSE ≈ 2.05 mm
- `validate_PCW2()` — static RC shear-wall cyclic test (Fig. 15), RMSE ≈ 0.11 mm
- `compare()` — correlation / R² between the two validation datasets

These are research scripts with experiment-specific hardcoded paths and crop boxes (e.g. the commented-out `crop = (...)` alternatives inside `validate_PCW2()`). To reproduce a specific result, uncomment the corresponding call in the `if __name__ == '__main__':` block at the bottom of the file, then run:

```bash
python validation.py
```

Outputs (detected positions, displacement curves, comparison plots) are written under `logs/disp/` and `logs/detected_disp/`.

### 5.4 `depth_measurement.py` — out-of-plane (depth) validation (Section 5.2)

`Depth_Estimator` extends `Validator` and implements the similar-triangles depth estimate (Eq. 26 in the paper) from a target's change in apparent size. `main()` at the bottom of the file reproduces the frontal-vs-angled-camera comparison (Fig. 17b):

```bash
python depth_measurement.py
```

When the angled camera is used, frames are first rectified with `utils/correct_perspective.py` (pass `perspective=True` to `read_video`).

### 5.5 `target.py` / `utils/correct_perspective.py` — perspective correction

Detects the four corners of the chessboard target via edge detection + Hough lines, then computes and applies a `cv2` perspective-transform matrix to rectify an angled camera view to a frontal-equivalent one. Can be run standalone for inspection:

```bash
python target.py
```

### 5.6 `plot_template.py` — shared plotting helper

A matplotlib wrapper (`plot_template(...)`) used by `validation.py` and `depth_measurement.py` to produce the paper's line-plot style (Times New Roman, inward tick marks, configurable legends/limits). Run it directly to see example output:

```bash
python plot_template.py
```

### 5.7 `test.py`, `util.py` — misc helpers

`test.py` is a short script that scans `logs/positions.txt` for frames where detection failed (`None`) and records their indices to `numbers.txt`. `util.py` holds standalone helpers used during earlier stages of the project (video→frame extraction, OCR-based position parsing via `pytesseract`); neither is required for the main detection/tracking/validation pipeline.

## 6. Retraining or adapting to a new target

This repository ships the trained weights used in the paper, so no training is required for inference or for reproducing the validation results above. If you want to adapt the framework to a different target or dataset:

- **Deep SORT appearance model** can be retrained with `nets/deep/train_deepsort.py` on a Market1501-style folder of cropped target images.
- **The Y-BWA-W detector's own training script** is not included in this folder (this repo focuses on inference/validation). If you'd like to retrain or fine-tune the detector, feel free to reach out (see Contact) — I'm happy to share the training pipeline and discuss adapting it to other target types, as noted in the paper's future-work discussion.

## 7. Key results (from the paper)

| Model | AP (%) | Precision (%) | Recall (%) |
|---|---|---|---|
| YOLO (baseline) | 93.20 | 97.31 | 92.71 |
| Y-SE / Y-CBAM / Y-CA / Y-ECA | 93.05–93.25 | 97.19–97.38 | 90.99–91.35 |
| **Y-BWA (proposed)** | **93.33** | 97.44 | 91.73 |

| Validation experiment | RMSE | Notes |
|---|---|---|
| Dynamic shaking-table test (in-plane) | 2.05 mm | < 1-pixel physical displacement (4.305 mm) |
| Static RC shear-wall test (in-plane) | 0.11 mm | < 1-pixel physical displacement (1.8 mm) |
| Depth test, frontal camera (out-of-plane) | 6.98 mm | ≈ 2-pixel level |
| Depth test, angled camera (out-of-plane) | 7.50 mm | ≈ 2-pixel level |

Real-time throughput: ~35 FPS on an NVIDIA RTX 4090.

## 8. References

This implementation builds on two open-source codebases:

- Bubbliiiing, [**yolov5-pytorch**](https://github.com/bubbliiiing/yolov5-pytorch) — the YOLOv5 (PyTorch) detector implementation this project's `yolo.py`/`predict.py`/`nets/` are built from, extended here with the BWA attention module and WNMS.
- ZQPei, [**deep_sort_pytorch**](https://github.com/ZQPei/deep_sort_pytorch) — the Deep SORT tracker implementation this project's `nets/deep_sort.py`, `nets/sort/`, and `nets/deep/` are built from.

And on the following methods/papers (full citations in the accepted manuscript):

- J. Redmon, S. Divvala, R. Girshick, A. Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR, 2016.
- J. Redmon, A. Farhadi, *YOLO9000: Better, Faster, Stronger*, CVPR, 2017.
- J. Redmon, A. Farhadi, *YOLOv3: An Incremental Improvement*, arXiv:1804.02767, 2018.
- C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, I.-H. Yeh, *CSPNet: A New Backbone that can Enhance Learning Capability of CNN*, CVPRW, 2020.
- K. He, X. Zhang, S. Ren, J. Sun, *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition*, IEEE TPAMI, 2015.
- K. Wang, J. H. Liew, Y. Zou, D. Zhou, J. Feng, *PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment*, ICCV, 2019.
- J. Hu, L. Shen, G. Sun, *Squeeze-and-Excitation Networks*, CVPR, 2018. (SE attention baseline)
- Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu, *ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks*, CVPR, 2020. (ECA attention baseline)
- S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, *CBAM: Convolutional Block Attention Module*, ECCV, 2018. (CBAM attention baseline)
- Q. Hou, D. Zhou, J. Feng, *Coordinate Attention for Efficient Mobile Network Design*, CVPR, 2021. (CA attention baseline)
- M. B. Muhammad, M. Yeasin, *Eigen-CAM: Class Activation Map using Principal Components*, IJCNN, 2020. (used for the interpretability heatmaps)
- J. P. Lewis, *Fast Normalized Cross-Correlation*, Vision Interface, 1995. (template matching used for initial dataset annotation)
- H. Zhang, N. Wang, *On the Stability of Video Detection and Tracking*, arXiv:1611.06467, 2017. (motivates the WNMS jitter-mitigation approach)
- N. Wojke, A. Bewley, D. Paulus, *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP, 2017. (Deep SORT)
- R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, J. Basic Eng., 1960. (Kalman filter)
- P. C. Mahalanobis, *On Tests and Measures of Group Divergence*, J. Asiat. Soc. Bengal, 1930. (Mahalanobis distance)
- H. W. Kuhn, *The Hungarian Method for the Assignment Problem*, Naval Research Logistics Quarterly, 1955. (data-association assignment)
- A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS, 2019.

## 9. Citation

```bibtex
@article{zhang2026vision,
  title   = {Vision-based Real-time Measurement of Structural Translational Motion Using Deep Learning and Object Tracking Methods},
  author  = {Zhang, Haoyou and Cheng, Xiaowei and Li, Yi and Guan, Hong},
  journal = {Applied Mathematical Modelling},
  year    = {2026},
  doi     = {10.1016/j.apm.2026.117187}
}
```

If you use the dataset itself, please also cite:

```bibtex
@dataset{zhang2026dataset,
  title     = {Vision-based Real-time Structural Displacement Measurement Using Deep Learning --- Dataset},
  author    = {Zhang, Haoyou and Cheng, Xiaowei and Li, Yi and Guan, Hong},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21400638}
}
```

## 10. Contact

Haoyou Zhang — **haoyou.zhang@marquette.edu**

Whether it's a bug, a question about adapting the method to your own targets/tests, or just a discussion about vision-based SHM — feel free to reach out. I enjoy discussing this work.
