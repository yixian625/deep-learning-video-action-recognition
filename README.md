# Video Classification with UCF50

This project implements a video classification pipeline using the UCF50 dataset (originally written for HMDB51). It leverages a Long-term Recurrent Convolutional Network (LRCN) model that extracts spatial features from individual video frames via a ResNet backbone and learns temporal dynamics through an LSTM. The project includes scripts for preprocessing, training, and testing the model.

---

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
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
- **Run the training loop** while tracking loss and accuracy, saving the best model weights.

---

## Testing and Evaluation

### Step 2: Run Testing

- **Configure Testing Parameters:**  
  Update the `--ckpt` argument in your testing script (e.g., `test.sh`) to point to the saved best model weights generated during training.

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

---

## Customization and Hyperparameters

You can modify several parameters to experiment with different settings:

### Data Parameters

- `--frame_dir`: Path to your preprocessed frames.
- `--fr_per_vid`: Number of frames to sample per video.

### Model Parameters

- `--model_type`: Choose between `'lrcn'` (default) or other supported models.
- `--cnn_backbone`: Options include `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`.
- `--rnn_hidden_size` and `--rnn_n_layers`: Configure the LSTM network.

### Training Parameters

- `--batch_size`, `--learning_rate`, `--n_epochs`, and `--dropout` control the training dynamics.
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

---

## Fixes

1. **Data leak in train/val/test split (fixed):** `dataset_split` now uses `StratifiedGroupKFold`. `get_video_group` gives every clip a group ID of the form `<class>_g<group>`, taken from its UCF50 name, and the split assigns each group whole to train, val, or test while keeping the class balance. With the default 80/10/10 split, no group or clip appears in more than one split, and all 50 classes appear in each. Because whole groups are assigned, the per-class ratios are approximate. Expect lower (but honest) accuracy than with the clip-level split.
2. **Frames not loaded in temporal order (fixed):** `VideoDataset.__getitem__` now sorts frame paths by their numeric index (`frame0, frame1, ..., frame9, frame10, ...`) before taking the first `fr_per_vid` frames.
3. **LSTM steps over the batch instead of over time (fixed):** `LRCN` now builds its LSTM with `batch_first=True`. `forward` extracts CNN features for all `batch_size x time_steps` frames in one pass, reshapes them to `(batch_size, time_steps, features)`, and runs the LSTM once over each video's frame sequence. A video's prediction no longer depends on the rest of the batch. Parameter names and shapes are unchanged, so old checkpoints still load, but models trained before this fix should be retrained.
