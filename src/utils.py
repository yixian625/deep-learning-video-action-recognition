"""
Module: utils.py

This module provides helper functions for video processing and data transformations
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Sampling a clip of consecutive (strided) frames from videos, for 3D CNNs.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing video-level (LRCN), clip-level (3D CNN) and optical flow data transforms for training and
      validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0
    frame_idx = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def get_clip_frames(vid, n_frames=32, stride=2):
    """
    Sample a clip of n_frames frames taken every `stride` frames from the middle of a video.

    Unlike get_frames, which spreads frames over the whole video, this keeps the frames close
    together in time, which is what 3D CNNs (and optical flow) expect. If the video is too short,
    the clip is padded by repeating its last frame, so every clip has exactly n_frames frames.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames in the clip.
        stride (int): Temporal stride between consecutive clip frames.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of n_frames frames (as numpy arrays in RGB format), or an empty
              list if the video cannot be read.
            - v_len (int): Total number of frames in the video, as reported by OpenCV.
    """
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return [], 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Center a window of (n_frames - 1) * stride + 1 frames in the video
    span = (n_frames - 1) * stride + 1
    start = max(0, (v_len - span) // 2)
    frame_idx = [start + i * stride for i in range(n_frames)]

    # Read sequentially and keep the frames at frame_idx (seeking is unreliable for some codecs)
    kept = {}
    for idx in range(frame_idx[-1] + 1):
        success, frame = v_cap.read()
        if not success:
            break
        if idx in frame_idx:
            kept[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    v_cap.release()
    if not kept:
        print("No frames found in video:", vid)
        return [], v_len

    # Pad with the last frame that could be read (short videos / overestimated frame counts)
    frames, last = [], None
    for idx in frame_idx:
        last = kept.get(idx, last)
        frames.append(last)
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    for idx, frame in enumerate(frames):
        print("processing")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f"frame{idx}.jpg")
        cv2.imwrite(path_to_frame, frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112;
    for 's3d' and 's3d_flow', clips are cropped to 224x224 (matching S3D's Kinetics-400 pretraining).
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn', '3dcnn', 's3d' or 's3d_flow').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    elif model == 's3d':
        h, w = 224, 224
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    elif model == 's3d_flow':
        # Flow (x, y) is already zero-centered and scaled to roughly [-1, 1] by FlowClipDataset
        h, w = 224, 224
        mean = [0.0, 0.0]
        std = [1.0, 1.0]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets (LRCN).

    The training transforms include data augmentation such as random horizontal flipping and random affine
    transformations, while the validation/test transforms consist solely of resizing, converting to float, and
    normalizing. The transforms operate on a whole video (a uint8 video tensor of shape (T, C, H, W)), so every random
    parameter (flip, translation) is drawn once per video and applied identically to all of its frames.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    train_transforms = v2.Compose([
        v2.Resize((height, width), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    val_test_transforms = v2.Compose([
        v2.Resize((height, width), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def compose_clip_transforms(height, width, mean, std, resize=(256, 256)):
    """
    Compose and return clip-level data transforms for 3D CNNs (e.g., S3D).

    The transforms operate on a whole clip (a uint8 video tensor of shape (T, C, H, W)), so every
    random parameter (crop position, flip) is drawn once per clip and applied identically to all of
    its frames. This keeps the motion between frames intact, which per-frame transforms would break.

    Args:
        height (int): Crop height.
        width (int): Crop width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        resize (tuple, optional): Size each frame is resized to before cropping. Default is (256, 256).

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Random crop and horizontal flip, then normalization.
            - val_test_transforms: Center crop, then normalization.
    """
    train_transforms = v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.RandomCrop((height, width)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    val_test_transforms = v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.CenterCrop((height, width)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


class FlowClipTransform:
    """
    Clip-level spatial transform for optical flow clips of shape (T, 2, H, W).

    Resizes, crops and (for training) randomly flips all flow fields of a clip identically. Unlike
    image transforms, it also updates the flow values: resizing scales the x and y flow by the
    horizontal and vertical resize factors, and a horizontal flip negates the x flow (motion to the
    right becomes motion to the left).

    Args:
        height (int): Crop height.
        width (int): Crop width.
        mean (list): Mean of the (x, y) flow channels for normalization.
        std (list): Standard deviation of the (x, y) flow channels for normalization.
        resize (tuple, optional): Size each flow field is resized to before cropping. Default is (256, 256).
        train (bool, optional): If True, random crop and horizontal flip; else center crop. Default is True.
    """
    def __init__(self, height, width, mean, std, resize=(256, 256), train=True):
        self.height, self.width = height, width
        self.mean = torch.tensor(mean).view(1, 2, 1, 1)
        self.std = torch.tensor(std).view(1, 2, 1, 1)
        self.resize = resize
        self.train = train

    def __call__(self, flow):
        h0, w0 = flow.shape[-2:]
        flow = torch.nn.functional.interpolate(flow, size=self.resize, mode='bilinear', align_corners=False)
        scale = torch.tensor([self.resize[1] / w0, self.resize[0] / h0]).view(1, 2, 1, 1)
        flow = flow * scale

        max_top, max_left = self.resize[0] - self.height, self.resize[1] - self.width
        if self.train:
            top = int(torch.randint(0, max_top + 1, (1,)))
            left = int(torch.randint(0, max_left + 1, (1,)))
        else:
            top, left = max_top // 2, max_left // 2
        flow = flow[..., top:top + self.height, left:left + self.width]

        if self.train and torch.rand(1).item() < 0.5:
            flow = flow.flip(-1)
            flow[:, 0] = -flow[:, 0]
        return (flow - self.mean) / self.std


def compose_flow_transforms(height, width, mean, std, resize=(256, 256)):
    """
    Compose and return clip-level transforms for optical flow clips (two-stream flow stream).

    Matches compose_clip_transforms (resize to 256x256, random or center 224x224 crop, horizontal
    flip for training) while keeping the flow values consistent with the geometry (see FlowClipTransform).

    Args:
        height (int): Crop height.
        width (int): Crop width.
        mean (list): Mean of the (x, y) flow channels for normalization.
        std (list): Standard deviation of the (x, y) flow channels for normalization.
        resize (tuple, optional): Size each flow field is resized to before cropping. Default is (256, 256).

    Returns:
        tuple: (train_transforms, val_test_transforms)
    """
    return (FlowClipTransform(height, width, mean, std, resize, train=True),
            FlowClipTransform(height, width, mean, std, resize, train=False))


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn', num_workers=0):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int, optional): Number of DataLoader worker processes. Default is 0 (load in the main process).

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=num_workers)
    val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn', num_workers=0):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int, optional): Number of DataLoader worker processes. Default is 0 (load in the main process).

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    test_dl = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=num_workers)
    dataloaders = {'test': test_dl}
    return dataloaders
