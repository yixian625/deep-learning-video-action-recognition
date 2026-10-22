# Video Classification with UCF50

This project implements a video classification pipeline using the UCF50 dataset (originally written for HMDB51). It leverages a Long-term Recurrent Convolutional Network (LRCN) model that extracts spatial features from individual video frames via a ResNet backbone and learns temporal dynamics through an LSTM. It also supports an S3D 3D CNN pretrained on Kinetics-400 and a two-stream (RGB + optical flow) S3D model (see [3D CNN (S3D)](#3d-cnn-s3d)). The project includes scripts for preprocessing, training, and testing the models.

---

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [3D CNN (S3D)](#3d-cnn-s3d)
- [Project Structure](#project-structure)
- [Customization and Hyperparameters](#customization-and-hyperparameters)

---

## Dataset Preparation

### Step 0: Download and Unzip Dataset

1. **Download Dataset:**  
   Download the UCF50 dataset (`UCF50.rar`) from [UCF CRCV](https://www.crcv.ucf.edu/data/UCF50.php). It contains 6,681 videos of 50 human action classes.

2. **Unzip and Organize:**  
   Extract the archive into the `data/` folder (which is git-ignored). The expected folder structure is:
   
        - data
            - UCF50
                - BaseballPitch
                    - v_BaseballPitch_g01_c01.avi
                    - ...
                - Basketball
                ... ... ... ...
                - YoYo

Each subdirectory represents a different action class. Videos are named `v_<class>_g<group>_c<clip>.avi`: every class has 25+ groups, and all clips in a group are cut from the same source video (same actor and scene).

---

## Environment Setup

1. **Python Version:**  
This project requires Python 3.7 or higher.

2. **Dependencies:**  
Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

# Key Libraries

- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**

## Hardware Requirements

A CUDA-enabled GPU is recommended for training. The code automatically detects GPU availability.

---

## Preprocessing and Frame Extraction

Before training, the raw video files must be converted into frame sequences. The preprocessing module includes functions for:

### Uniform Frame Sampling

- The `get_frames` function uses OpenCV to sample a fixed number of frames per video.

### Saving Frames to Disk

- The `store_frames` function writes the extracted frames as JPEG images.

`preprocess.py` uses these functions to convert every video into a folder of extracted frames:

```bash
python preprocess.py --video_dir data/UCF50 --frame_dir data/UCF50_frames --n_frames 16
```

The resulting folder structure mirrors the original dataset structure:

        - data
            - UCF50_frames
                - BaseballPitch
                    - v_BaseballPitch_g01_c01
                        - frame0.jpg
                        - frame1.jpg
                        - ...


---

## Training the Model

### Step 1: Run Training

#### Configure Training Parameters

The training is managed via a bash script (e.g., `train.sh`) that calls the main training module.  
**Important:** Update the `--frame_dir` argument in the script to point to the directory where your preprocessed frame data is stored. You can also adjust other parameters (e.g., number of frames per video, batch size, learning rate) to see how they affect the experiment.

#### Run the Training Script

Execute the training script from your terminal:

```bash
bash train.sh
```

## During Training, the Script Will:

- **Load the frame dataset.**
- **Split the dataset** into training, validation, and test sets (80/10/10 by default) using stratified group sampling, so all clips from the same source video stay in the same split.
- **Apply data augmentation** techniques (resizing, random flips, affine transformations).
- **Create custom PyTorch Datasets and DataLoaders.**
- **Initialize the LRCN model** using a specified ResNet backbone.
- **Set up the loss function, optimizer, and learning rate scheduler.**
- **Run the training loop** while tracking loss and accuracy, saving the best model weights and the dataset splits to `models/<model_type>/` (override with `--output_dir`).

---

## Testing and Evaluation

### Step 2: Run Testing

- **Configure Testing Parameters:**  
  Update the `--ckpt` argument in your testing script (e.g., `test.sh`) to point to the saved best model weights generated during training (by default `models/lrcn/best_model_wts.pt`). The splits are loaded from the same folder.

- **Run the Testing Script:**  
  Execute the testing script from your terminal:
  
```bash
bash test.sh
```

## Testing Script Overview

The testing script will:

- **Load the dataset splits** (previously saved during training).
- **Create a DataLoader for the test set.**
- **Load the trained model checkpoint.**
- **Evaluate the model** on the test data by computing overall accuracy, generating classification reports, and optionally producing confusion matrices.
- **Save per-video class probabilities** to `test_predictions.npz` next to the checkpoint (see [Two-stream](#two-stream-rgb--optical-flow)).

---

## 3D CNN (S3D)

S3D is a separable 3D CNN from the I3D family, built from torchvision with Kinetics-400 pretrained weights (`s3d-d76dad2f.pth`, 32 MB, downloaded automatically from `download.pytorch.org` on the first run). Unlike LRCN, it takes a short clip of nearby frames instead of frames spread over the whole video.

1. **Extract dense clips** of 32 consecutive frames, taking every 2nd frame from the middle of each video. This is about 12.5-15 fps, close to S3D's 15 fps pretraining, and takes about 4.7 GB. Videos shorter than 63 frames are padded by repeating their last frame.

   ```bash
   python preprocess.py --video_dir data/UCF50 --frame_dir data/UCF50_clips --n_frames 32 --sampling dense --stride 2
   ```

2. **Train and evaluate:**

   ```bash
   bash train_s3d.sh
   bash test_s3d.sh
   ```

   These use `--model_type s3d --fr_per_vid 32 --batch_size 8 --amp --num_workers 6`. With mixed precision (`--amp`), a batch of 8 clips at 224x224 uses about 4.4 GB of GPU memory.

**Clip-level augmentation:** `ClipDataset` applies `compose_clip_transforms` (resize to 256x256, random 224x224 crop, horizontal flip) to the whole clip at once. Every frame of a clip gets the same crop and flip, which keeps the motion between frames intact.

### Results

Results are on UCF50 with the group-aware 80/10/10 split (5,343 / 668 / 670 clips), from a single run (`train_s3d.sh`, no fixed seed).

| Hyperparameter | Value |
|---|---|
| Model | S3D, Kinetics-400 pretrained, new 50-class 1x1x1 conv head, dropout 0.2 |
| Input | 32 frames at stride 2, 224x224 crops |
| Optimizer | Adam, lr 1e-4, no weight decay, cross-entropy (`reduction='sum'`) |
| LR schedule | `ReduceLROnPlateau` on val loss (factor 0.5, patience 5); lr dropped to 5e-5 after epoch 10 |
| Batch size / epochs | 8 / 20 (no early stopping), mixed precision |
| Checkpoint | best val accuracy |

| Metric | Value |
|---|---|
| Best val accuracy | 91.8% (epoch 12) |
| **Test accuracy** | **93.9%** |
| Test macro F1 | 0.935 |

The weakest test classes are Nunchucks (8/16), Fencing (5/9) and PullUps (8/12). With lr 1e-4, validation accuracy swung between 79% and 88% in epochs 5-10, and the model only improved after the lr dropped to 5e-5. That suggests starting future runs at a lower lr.

### Two-stream (RGB + optical flow)

A second S3D stream is trained on optical flow, and its predictions are late-fused with the RGB stream.

1. **Compute flow:** run RAFT-small (torchvision, `raft_small_C_T_V2-01064c6d.pth`, 3.8 MB, downloaded on the first run) on the 31 consecutive frame pairs of each clip in `data/UCF50_clips`, so the flow lines up in time with the RGB clips. The x and y flow, clipped to ±20 px, are stored as grayscale JPEGs `flow_x_{i}.jpg` / `flow_y_{i}.jpg` in `data/UCF50_flow`.

   ```bash
   python compute_flow.py --clip_dir data/UCF50_clips --flow_dir data/UCF50_flow --amp
   ```

2. **Train and evaluate the flow stream** with `--model_type s3d_flow --fr_per_vid 32`. That's 31 flow fields plus the last one repeated: an odd clip length of 31 makes cuDNN about 2.8x slower. Both streams use separate learning rates for the new classification layer (`--learning_rate 5e-4`) and the pretrained backbone (`--backbone_lr`), plus early stopping: training stops once validation loss hasn't decreased for 10 epochs (`--early_stopping_patience 10`, `--n_epochs 50` maximum). The flow stream differs from the RGB pretraining much more, so it uses a 2x higher backbone lr (1e-4 vs 5e-5 for RGB). It also uses BatchNorm momentum 0.01 (`--bn_momentum`; torchvision's S3D default of 0.001 would keep the RGB running statistics for several epochs). W&B runs are named `s3d-two-flow-train`, `s3d-two-flow-test` and `s3d-two-fusion` (`--run_name`).

   ```bash
   bash train_s3d_flow.sh
   bash test_s3d_flow.sh
   ```

   The RGB stream is retrained with the same schedule (separate lrs and early stopping; backbone lr 5e-5, default BatchNorm momentum) into `models/s3d_two_rgb/`, with W&B runs `s3d-two-rgb-train` / `s3d-two-rgb-test`:

   ```bash
   bash train_s3d_rgb.sh
   bash test_s3d_rgb.sh
   ```

3. **Fuse:** `fuse.py` matches both streams' `test_predictions.npz` (`models/s3d_two_rgb/` and `models/s3d_flow/`) by video and averages their class probabilities with equal weights (the weights are not tuned on the test set). It reports RGB, flow and fused test accuracy, and logs them to W&B.

   ```bash
   bash fuse_two_stream.sh
   ```

   To choose the weights on the **validation** set instead, `fuse_two_stream_val_weighted.sh` evaluates both streams on the validation split (`run.py --eval_split val` writes `val_predictions.npz`). `fuse.py --val_preds` then grid-searches the flow weight (step 0.05) for the best fused validation accuracy, and applies it to the test predictions.

   ```bash
   bash fuse_two_stream_val_weighted.sh
   ```

**Implementation notes**
- **Model:** `build_s3d(..., in_channels=2)` replaces S3D's first convolution and initializes it with the average of the pretrained RGB filters (cross-modality initialization, as in TSN/I3D). The rest of the network keeps its Kinetics-400 weights.
- **Augmentation:** `FlowClipDataset` and `FlowClipTransform` apply the same resize, crop and flip as the RGB stream, and keep the flow values consistent with the geometry. Resizing scales the x/y flow by the resize factors, and a horizontal flip negates the x flow.
- **Same split:** the split is computed from the video names, so both streams get the identical train/val/test split.

### Two-stream results

Results are on the same split and test set as [Results](#results), from single runs without a fixed seed. Both streams use Adam with head lr 5e-4, `ReduceLROnPlateau` (factor 0.5, patience 5), early stopping (patience 10, max 50 epochs), batch size 8 and mixed precision. The checkpoint is the one with the best val accuracy.

| Model | Backbone lr | Epochs | Best val acc | **Test acc** | Test macro F1 |
|---|---|---|---|---|---|
| S3D RGB, original run (`train_s3d.sh`, single lr 1e-4) | 1e-4 | 20 (fixed) | 91.8% | 93.9% | 0.935 |
| **S3D RGB stream** (`train_s3d_rgb.sh`) | 5e-5 | 23 (early stop) | 95.1% | **94.9%** | **0.949** |
| S3D flow stream (`train_s3d_flow.sh`, BN momentum 0.01) | 1e-4 | 49 (early stop) | 82.0% | 75.7% | 0.742 |
| Fusion, equal weights | | | | 92.5% | |
| Fusion, weights chosen on val (RGB 0.9 / flow 0.1) | | | 95.5% | 94.9% | |

**Fusion does not improve on RGB alone.** This is how the two streams' answers compare on the 670 test videos:

| | Flow correct | Flow wrong |
|---|---|---|
| **RGB correct** | 493 | 143 |
| **RGB wrong** | 14 | 20 |

- **Fusion can only fix the 14 videos that flow gets right and RGB gets wrong.** Averaging the two models' predictions can only turn a wrong RGB answer into a right one if flow has that video right. The 20 videos that both models get wrong can't be fixed. So even a perfect way of combining the two models would reach at most 97.0% (650 of 670 videos), 2.1 points above RGB alone.
- **Averaging with equal weights ruins more videos than it fixes.** There are 143 videos where RGB is right and flow is wrong, and averaging changed 27 of them to a wrong answer. It fixed only 11 of the 14. The net result is 16 fewer correct videos, so test accuracy drops from 94.9% to 92.5%. Flow does this damage partly because it's about as confident when it's wrong as RGB is: its top-class probability averages 0.77 on its wrong answers, versus 0.72 for RGB's wrong answers.
- **Choosing the weights on the validation set** gives flow only 10% of the weight. That improves validation accuracy by 3 videos, but doesn't change a single test prediction, so the fused test accuracy equals RGB's 94.9%.

The best model is the RGB stream (94.9% test). The flow stream is likely held back by its RGB-pretrained initialization: published two-stream gains, e.g. I3D, use flow streams pretrained on Kinetics flow. It is also limited by overfitting (train loss ~0.02 vs val loss ~0.9) and a single 2-second flow window per video.

---

## Customization and Hyperparameters

You can modify several parameters to experiment with different settings:

### Data Parameters

- `--frame_dir`: Path to your preprocessed frames.
- `--fr_per_vid`: Number of frames to sample per video.

### Model Parameters

- `--model_type`: Choose between `'lrcn'` (default), `'s3d'` (RGB) and `'s3d_flow'` (optical flow).
- `--cnn_backbone`: Options include `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`.
- `--rnn_hidden_size` and `--rnn_n_layers`: Configure the LSTM network.

### Training Parameters

- `--batch_size`, `--learning_rate`, `--n_epochs`, and `--dropout` control the training dynamics.
- `--amp` enables mixed precision and `--num_workers` sets the number of DataLoader worker processes.
- `--backbone_lr` sets a separate (usually smaller) learning rate for the pretrained backbone; `--learning_rate` then applies to the new head only (S3D: final 1x1x1 conv; LRCN: LSTM and final layer).
- `--early_stopping_patience N` stops training after N epochs without a decrease in validation loss (`--n_epochs` is then the maximum).
- `--run_name` sets the W&B run name (default `<model_type>-<mode>`).
- `--bn_momentum` (S3D only) sets the momentum of the BatchNorm running statistics (default 0.001).
- `--train_size` and `--test_size` determine dataset splits.

By tweaking these parameters, you can study their impact on model performance and experiment with different network configurations.

---

## Summary of Steps

- **Step 0: Dataset Preparation**  
  Download UCF50, extract it to `data/UCF50`, and run `preprocess.py` to extract frames into `data/UCF50_frames`.

- **Step 1: Run Training**  
  Execute `train.sh` after configuring the `--frame_dir` and other hyperparameters to train the model.

- **Step 2: Run Testing**  
  Execute `test.sh` after updating the `--ckpt` argument to point to the best model checkpoint to evaluate the model.

Happy Training!

---

## Issues in Original Implementation

1. **Data leak in train/val/test split:** `dataset_split` in `video_datasets.py` stratifies at the clip level. HMDB51 cuts several clips from each source video (e.g. `..._goo_0`, `..._goo_1`), so clips of the same actor and scene can end up in both train and val/test. This inflates validation and test accuracy. The fix is to use the official HMDB51 split files, or to split by source-video group (e.g. `GroupShuffleSplit`).
2. **Frames not loaded in temporal order:** `VideoDataset.__getitem__` uses unsorted `glob`, and the names `frame0.jpg ... frame16.jpg` sort as strings (`frame10` comes before `frame2`). The LSTM therefore sees frames in a scrambled order. The fix is to sort frame paths by their numeric index.
3. **LSTM steps over the batch instead of over time:** `LRCN.forward` loops over frames and feeds each frame's features to the LSTM as a `(batch, 1, features)` tensor, but the LSTM is created with the default `batch_first=False`, so it reads that as `(time, batch, features)`. At every frame the LSTM therefore steps through the videos in the batch as if they were time steps, and carries that state to the next frame. Each video's prediction depends on the other videos in its batch and their order, and the model does not learn per-video temporal dynamics. The fix is to pass each video's frame features as one sequence with `batch_first=True`.
4. **Augmentation applied per frame instead of per video:** `compose_data_transforms` builds `RandomHorizontalFlip` and `RandomAffine(translate=0.1)` transforms, and `VideoDataset` applies them to each frame separately. Within one video, some frames are mirrored and others not, and every frame is shifted differently, so the LSTM sees fake motion (the person jumping between sides of the image, the scene shaking). The fix is to draw the random flip and shift once per video and apply them to all of its frames.

---

## Fixes

1. **Data leak in train/val/test split (fixed):** `dataset_split` now uses `StratifiedGroupKFold`. `get_video_group` gives every clip a group ID of the form `<class>_g<group>`, taken from its UCF50 name, and the split assigns each group whole to train, val, or test while keeping the class balance. With the default 80/10/10 split, no group or clip appears in more than one split, and all 50 classes appear in each. Because whole groups are assigned, the per-class ratios are approximate. Expect lower (but honest) accuracy than with the clip-level split.
2. **Frames not loaded in temporal order (fixed):** `VideoDataset.__getitem__` now sorts frame paths by their numeric index (`frame0, frame1, ..., frame9, frame10, ...`) before taking the first `fr_per_vid` frames.
3. **LSTM steps over the batch instead of over time (fixed):** `LRCN` now builds its LSTM with `batch_first=True`. `forward` extracts CNN features for all `batch_size x time_steps` frames in one pass, reshapes them to `(batch_size, time_steps, features)`, and runs the LSTM once over each video's frame sequence. A video's prediction no longer depends on the rest of the batch. Parameter names and shapes are unchanged, so old checkpoints still load, but models trained before this fix should be retrained.
4. **Augmentation applied per frame instead of per video (fixed):** `compose_data_transforms` now uses `torchvision.transforms.v2`, and `VideoDataset` loads a video's frames as one uint8 video tensor `(T, C, H, W)` and transforms it in a single call. The random flip and translation are therefore drawn once per video and are identical for all of its frames. Validation/test inputs are unchanged apart from negligible resize differences (mean absolute difference of about 0 after normalization).
