# Video Action Recognition on UCF50

This project classifies human actions in videos from the UCF50 dataset (50 classes, 6,681 videos) with three models (results in [section 6](#6-results-comparison)):

| Model | Input | Pretrained on |
|---|---|---|
| [LRCN](#3-lrcn-resnet--lstm) (ResNet-34 + LSTM) | 16 frames spread over the whole video | ImageNet |
| [S3D](#4-3d-cnn-s3d) (3D CNN) | 32 frames from a 2-second clip | Kinetics-400 |
| [Two-stream S3D](#5-two-stream-s3d-rgb--optical-flow) (RGB + optical flow) | the S3D clip and its optical flow | Kinetics-400 |

The code started as an LRCN pipeline for HMDB51. It was moved to UCF50, four bugs in the original implementation were fixed (see [3.4](#34-issues-in-the-original-implementation)), and the S3D and two-stream models were added.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [How Training and Evaluation Work](#2-how-training-and-evaluation-work)
3. [LRCN (ResNet + LSTM)](#3-lrcn-resnet--lstm)
4. [3D CNN (S3D)](#4-3d-cnn-s3d)
5. [Two-Stream S3D (RGB + Optical Flow)](#5-two-stream-s3d-rgb--optical-flow)
6. [Results Comparison](#6-results-comparison)
7. [Development](#7-development)

---

## 1. Getting Started

### 1.1 Environment setup

The project uses Python 3.13 and the CUDA 13.0 build of PyTorch, which needs an NVIDIA GPU whose driver supports CUDA 13.0. `requirements.txt` points pip to PyTorch's CUDA 13.0 package index, so the GPU build is installed:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash); on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation are logged to [Weights & Biases](https://wandb.ai) in the project `video-action-lrcn`. Log in once per machine with `wandb login`, or turn logging off for a run with `--wandb_mode disabled`.

### 1.2 Dataset

Download `UCF50.rar` from [UCF CRCV](https://www.crcv.ucf.edu/data/UCF50.php) and extract it into `data/` (git-ignored):

        - data
            - UCF50
                - BaseballPitch
                    - v_BaseballPitch_g01_c01.avi
                    - ...
                - Basketball
                ... ... ... ...
                - YoYo

Each folder is one action class. Videos are named `v_<class>_g<group>_c<clip>.avi`. All clips in a group are cut from the same source video (same actor and scene), and each class has 25+ groups.

### 1.3 Project structure

```
src/                         Python source
  preprocess.py              extract frames (uniform for LRCN, dense clips for S3D)
  compute_flow.py            RAFT optical flow for the two-stream flow stream
  run.py                     main entry point: train or evaluate a model
  train.py                   training loop (early stopping, W&B logging)
  evaluate.py                evaluation and classification report
  fuse.py                    late fusion of two streams' predictions
  models.py                  LRCN, S3D (RGB and flow) and the model factory
  video_datasets.py          datasets, group-aware train/val/test split, collate functions
  utils.py                   frame sampling, transforms and DataLoaders
bash/                        training, testing and fusion scripts
test/                        regression tests for the fixed bugs
pylint/                      pylint settings (pylintrc), report script and latest report
data/                        datasets, extracted frames, clips and flow (git-ignored)
models/                      checkpoints, splits and predictions per model (git-ignored)
pytest.ini                   pytest settings
```

The scripts in `bash/` change to the repo root first, so they can be run from any folder (e.g. `bash bash/train.sh`). They call `python`, so activate the `.venv` first.

---

## 2. How Training and Evaluation Work

All three models are trained and evaluated with `src/run.py` (`--mode train` or `--mode eval`). The `bash/` scripts call it with each model's settings.

**The train/val/test split** is 80/10/10 (5,343 / 668 / 670 clips) and is grouped by source video. Every group (`<class>_g<group>`) goes entirely into train, val or test, so the models are always tested on source videos they have never seen. Classes stay balanced across the three sets. The split is fixed by a seed, and it is computed from the video names, so all models get the identical split.

**Train mode:**
- Saves the split as `splits.npy` in `models/<model>/` (change with `--output_dir`).
- Trains with Adam and `ReduceLROnPlateau`. When validation loss stops improving for 6 epochs, the scheduler halves the learning rate, and training restarts from the best weights so far.
- Saves the checkpoint with the best validation accuracy as `best_model_wts.pt`.
- Optionally stops early once validation loss hasn't decreased for a set number of epochs (`--early_stopping_patience`).

**Eval mode:**
- Loads the checkpoint (`--ckpt`) and the `splits.npy` saved next to it, and evaluates the test split (or the validation split with `--eval_split val`).
- Saves each video's class probabilities to `<split>_predictions.npz` next to the checkpoint, which the two-stream fusion uses.
- Logs the accuracy, a per-class report and a confusion matrix to W&B.

---

## 3. LRCN (ResNet + LSTM)

### 3.1 Overview

LRCN extracts features from each frame with an ImageNet-pretrained ResNet-34, and an LSTM models how they change over time. This is the original model of the repository.

### 3.2 Preprocessing

Extract 16 frames spread evenly over each video. The script stores 17 frames (`frame0.jpg` ... `frame16.jpg`); the loader uses the first 16. This takes about 3 minutes and 2.5 GB.

```bash
python src/preprocess.py --video_dir data/UCF50 --frame_dir data/UCF50_frames --n_frames 16
```

### 3.3 Train & test

```bash
bash bash/train.sh     # checkpoint and split in models/lrcn/
bash bash/test.sh
```

Training takes about 3.5 hours (30 epochs) on an RTX 3060 Laptop GPU.

### 3.4 Issues in the original implementation

The original LRCN code had four bugs. Each is fixed, and a regression test in `test/` fails if it comes back.

| # | Issue | Fix | Test |
|---|---|---|---|
| 1 | **Data leak:** the split was made per clip, so clips cut from the same source video ended up in both train and test, which inflated accuracy. | Split by source-video group (`StratifiedGroupKFold`), see [section 2](#2-how-training-and-evaluation-work). | `test_dataset_split.py` |
| 2 | **Frames out of order:** frames were loaded in string order (`frame10` before `frame2`), so the LSTM saw them scrambled. | Sort frames by their number. | `test_frame_order.py` |
| 3 | **LSTM ran over the batch, not over time:** each frame's features were passed as `(batch, 1, features)` to an LSTM that expects `(time, batch, features)`. It stepped through the videos of a batch instead of through each video's frames, so each prediction depended on the other videos in the batch. | Use `batch_first=True` and run the LSTM once over each video's frame sequence. | `test_lrcn.py` |
| 4 | **Augmentation per frame:** random flips and shifts were drawn separately for each frame, adding fake motion within a video. | Draw them once per video and apply them to all of its frames. | `test_augmentation.py` |

---

## 4. 3D CNN (S3D)

### 4.1 Overview

S3D is a 3D CNN from the I3D family, which learns motion directly from consecutive frames. It is built from torchvision with Kinetics-400 pretrained weights (`s3d-d76dad2f.pth`, 32 MB, downloaded on the first run), with a new 50-class output layer. S3D is used instead of the better-known I3D because the two are conceptually very similar (S3D is I3D's successor, splitting its 3D convolutions into separate spatial and temporal ones), but only S3D comes with Kinetics-400 weights in PyTorch's torchvision, while I3D would need third-party code and weights.

### 4.2 Preprocessing

Extract 32 frames from the middle of each video's timeline, taking every 2nd frame (e.g. frames 59, 61, ..., 121 of a 181-frame video). The full 320x240 frames are kept. That's about 2 seconds of video at 12.5–15 fps, close to the 15 fps S3D was pretrained on. Videos shorter than 63 frames are padded by repeating their last frame. This takes about 3 minutes and 4.2 GB.

The frames are saved as individual JPEGs, in the same layout as LRCN's (`data/UCF50_clips/<class>/<video>/frame0.jpg ... frame31.jpg`). The difference is which frames are kept: LRCN's 16 frames are spread over the whole video, while these 32 are close together in time. "Clip" refers to these 32 frames as a group: `ClipDataset` loads all of them and stacks them into one `(32, 3, H, W)` tensor, which is what S3D receives.

```bash
python src/preprocess.py --video_dir data/UCF50 --frame_dir data/UCF50_clips --n_frames 32 --sampling dense --stride 2
```

### 4.3 Train & test

```bash
bash bash/train_s3d.sh     # checkpoint and split in models/s3d/
bash bash/test_s3d.sh
```

Each frame is resized to 256x256 and cropped to 224x224: at a random position with a random horizontal flip for training (drawn once per clip, so all frames of a clip get the same augmentation), and at the center for validation and test. With mixed precision (`--amp`), a batch of 8 clips uses about 4.4 GB of GPU memory, and an epoch takes about 10 minutes.

---

## 5. Two-Stream S3D (RGB + Optical Flow)

### 5.1 Overview

Two S3D models are trained separately: one on the RGB clips, one on their optical flow (how each pixel moves between frames). Their predictions are averaged ("late fusion").

### 5.2 Preprocessing

This needs the S3D clips from [4.2](#42-preprocessing). The script runs RAFT-small (torchvision, `raft_small_C_T_V2-01064c6d.pth`, 3.8 MB, downloaded on the first run) on the 31 pairs of consecutive frames in each clip, so the flow lines up in time with the RGB clip. The horizontal and vertical flow, clipped to ±20 pixels, are saved as grayscale JPEGs (`flow_x_<i>.jpg`, `flow_y_<i>.jpg`). This takes about 30 minutes on the GPU and about 4 GB.

```bash
python src/compute_flow.py --clip_dir data/UCF50_clips --flow_dir data/UCF50_flow --amp
```

### 5.3 Train & test

Both streams use two learning rates: 5e-4 for the new output layer, and a smaller one for the pretrained layers. They also use early stopping (10 epochs without a lower validation loss, at most 50 epochs). The flow stream gets a 2× higher lr for its pretrained layers (1e-4 instead of 5e-5), because flow looks very different from the RGB video S3D was pretrained on.

```bash
bash bash/train_s3d_flow.sh     # flow stream: models/s3d_flow/
bash bash/test_s3d_flow.sh
bash bash/train_s3d_rgb.sh      # RGB stream: models/s3d_two_rgb/
bash bash/test_s3d_rgb.sh
```

Then fuse the two streams' test predictions:

```bash
bash bash/fuse_two_stream_val_weighted.sh   # weights chosen on the validation set
bash bash/fuse_two_stream.sh                # equal weights
```

The validation-weighted script first evaluates both streams on the validation split. It then tries flow weights from 0 to 1 in steps of 0.05, keeps the one with the best fused validation accuracy, and applies it to the test predictions. The weights are never tuned on the test set.

**Implementation notes**
- **Flow input layer:** S3D's first layer is replaced with a 2-channel one (x and y flow), initialized with the average of the pretrained RGB filters. The rest of the network keeps its Kinetics-400 weights.
- **Flow augmentation:** the flow gets the same resize, crop and flip as the RGB clips. Resizing scales the flow values by the same factor, and a horizontal flip negates the x flow.
- **BatchNorm momentum:** the flow stream uses 0.01 (`--bn_momentum`) instead of S3D's default 0.001, so the normalization statistics adapt to flow within an epoch.
- **Clip length:** flow clips are padded from 31 to 32 steps, because an odd length of 31 makes the GPU convolutions about 2.8× slower.

---

## 6. Results Comparison

All models are tested on the same 670 clips, from source videos none of them saw during training or validation. Each row is a single training run without a fixed seed.

| Model | Test accuracy | Macro F1 | Best val accuracy |
|---|---|---|---|
| LRCN: original settings, with the 4 bug fixes (`bash/train.sh`) | 86.6% | 0.861 | 82.6% |
| S3D (`bash/train_s3d.sh`) | 93.9% | 0.935 | 91.8% |
| **Two-stream S3D, fusion weights chosen on val (RGB 0.9 / flow 0.1)** (`bash/fuse_two_stream_val_weighted.sh`) | **94.9%** | **0.949** | 95.5%* |
| Two-stream S3D, equal fusion weights (`bash/fuse_two_stream.sh`) | 92.5% | 0.924 | |

\*The fused model's validation accuracy, at the weights the search picked.

**Note on training length:** the LRCN and S3D (RGB only) runs used a fixed number of epochs (30 and 20) without early stopping. Retraining them with the two-stream setup (early stopping, up to 50 epochs) would have taken several more hours each, so the available GPU time went to the two-stream models instead.

- **Kinetics-pretrained 3D CNN:** S3D beats LRCN by 7.3 points. Learning motion from consecutive frames, with pretraining on an action dataset, matters more than anything else here.
- **Two-stream:** the best two-stream model gives exactly the same test results as its RGB stream alone. Its gain over the S3D row comes from the RGB stream's better training settings (separate learning rates and early stopping), not from adding flow.

---

## 7. Development

### 7.1 Tests

`test/` contains one regression test file per fixed bug (see [3.4](#34-issues-in-the-original-implementation)):

```bash
python -m pytest
```

### 7.2 Linting

The code scores 10.00/10 with pylint. The settings, each with the reason for it, are in `pylint/pylintrc`, so pass that file explicitly:

```bash
python -m pylint --rcfile pylint/pylintrc src test
```

`bash pylint/lint_report.sh` runs the same check and saves the full report to `pylint/pylint_report.txt`.

### 7.3 Command-line options of `src/run.py`

**Data**
- `--frame_dir`: folder of extracted frames, clips or flow.
- `--fr_per_vid`: frames per video (LRCN) or per clip (S3D).
- `--train_size`, `--test_size`: split fractions (default 0.8 / 0.1; validation gets the rest).

**Model**
- `--model_type`: `lrcn` (default), `s3d` or `s3d_flow`.
- `--pretrained`: use pretrained weights (default on; pass `''` to turn off).
- `--cnn_backbone`: LRCN only: `resnet18`, `resnet34` (default), `resnet50` or `resnet101`. `resnet152` currently builds a ResNet-34.
- `--rnn_hidden_size`, `--rnn_n_layers`: LRCN only: LSTM size.
- `--dropout`: dropout before the output layer.
- `--bn_momentum`: S3D only: BatchNorm momentum (default 0.001).

**Training**
- `--batch_size`, `--learning_rate`, `--n_epochs`.
- `--backbone_lr`: separate learning rate for the pretrained layers. `--learning_rate` then applies only to the new layers: the output layer for S3D, and the LSTM and output layer for LRCN.
- `--early_stopping_patience N`: stop after N epochs without a decrease in validation loss (`--n_epochs` is then the maximum).
- `--amp`: mixed precision. `--num_workers`: number of data-loading processes.
- `--output_dir`: where to save the checkpoint and split (default `models/<model_type>`).

**Evaluation and logging**
- `--mode`: `train` or `eval`. `--ckpt`: checkpoint to evaluate. `--eval_split`: `test` (default) or `val`.
- `--run_name`: W&B run name (default `<model_type>-<mode>`). `--wandb_mode`: `online` (default), `offline` or `disabled`.
