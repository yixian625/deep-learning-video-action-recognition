"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes custom PyTorch Datasets for videos stored as directories
of frame images (per-frame transforms for LRCN, clip-level transforms for 3D CNNs), functions to
load the dataset from a directory structure, split the dataset using stratified group sampling,
and custom collate functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torch.nn.utils.rnn import pad_sequence

import os
import re
import glob
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def sorted_frame_paths(vid_dir, prefix=''):
    """
    Return the JPEG frame paths of a video directory sorted by frame index.

    Frames are named frame0.jpg, frame1.jpg, ...; sorting numerically keeps frame10 after frame9
    (plain string sorting would put it right after frame1).

    Args:
        vid_dir (str): Path to the video (frame) directory.
        prefix (str, optional): Only include files whose name starts with this prefix
            (e.g., 'flow_x_' for optical flow). Default is '' (all JPEG files).

    Returns:
        list: Frame paths in temporal order.
    """
    fr_paths = glob.glob(vid_dir + '/' + prefix + '*.jpg')
    return sorted(fr_paths, key=lambda p: int(re.search(r'(\d+)\.jpg$', p).group(1)))


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A video-level transform applied to a uint8 video tensor of shape
            (T, C, H, W) (e.g., resizing, normalization; see utils.compose_data_transforms).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths sorted by frame index (frame0, frame1, ..., frame10, ...)
        # and select up to fr_per_vid frames
        fr_paths = sorted_frame_paths(self.dataset[idx][0])
        fr_paths = fr_paths[:self.fpv]
        
        # Get the label associated with the video
        fr_label = self.dataset[idx][1]
        if not fr_paths:
            return [], fr_label

        # Load the frames as one uint8 video tensor of shape (T, C, H, W)
        fr_imgs = tv_tensors.Video(torch.stack([read_image(fr_path, mode=ImageReadMode.RGB) for fr_path in fr_paths]))

        # Apply transforms to the whole video at once, so random augmentations (flip, translation)
        # are identical for all of its frames
        if self.transforms:
            fr_imgs = self.transforms(fr_imgs)

        return fr_imgs.as_subclass(torch.Tensor), fr_label


class ClipDataset(Dataset):
    """
    PyTorch Dataset class for loading fixed-length clips for 3D CNNs (e.g., S3D).

    Each video is a directory of JPEG frames (e.g., created by preprocess.py with --sampling dense).
    The first n_frames frames are loaded in temporal order and padded by repeating the last frame
    if the video has fewer. Unlike VideoDataset, transforms are applied to the whole clip at once
    (see utils.compose_clip_transforms), so random augmentations are identical for all frames.

    Other input modalities for the same videos subclass this and override load_clip (see
    FlowClipDataset), keeping the same video keys so that predictions of the streams of a
    two-stream model can be matched per video.

    Args:
        vid_dataset (list): List of (video_directory_path, label) tuples.
        n_frames (int): Number of frames per clip.
        transforms (callable, optional): Clip-level transform applied to a uint8 video tensor of shape (T, C, H, W).
    """
    def __init__(self, vid_dataset, n_frames, transforms=None):
        self.dataset = vid_dataset
        self.n_frames = n_frames
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def load_clip(self, vid_dir):
        """
        Load the clip of a video as a uint8 video tensor of shape (n_frames, C, H, W).

        Args:
            vid_dir (str): Path to the video (frame) directory.

        Returns:
            tv_tensors.Video: The clip.
        """
        fr_paths = sorted_frame_paths(vid_dir)[:self.n_frames]
        if not fr_paths:
            raise RuntimeError('No frames found in {}'.format(vid_dir))
        frames = [read_image(fr_path, mode=ImageReadMode.RGB) for fr_path in fr_paths]
        frames += [frames[-1]] * (self.n_frames - len(frames))
        return tv_tensors.Video(torch.stack(frames))

    def __getitem__(self, idx):
        """
        Load the clip for the given index, apply the clip-level transforms, and return it with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (clip, label) where clip is a tensor of shape (T, C, H, W) and label is an integer.
        """
        vid_dir, label = self.dataset[idx]
        clip = self.load_clip(vid_dir)
        if self.transforms:
            clip = self.transforms(clip)
        return clip.as_subclass(torch.Tensor), label


class FlowClipDataset(ClipDataset):
    """
    PyTorch Dataset class for loading optical flow clips for the flow stream of a two-stream model.

    Each video is a directory of flow images created by compute_flow.py: flow_x_{i}.jpg and
    flow_y_{i}.jpg hold the horizontal and vertical flow between clip frames i and i + 1, quantized
    to [0, 255] over [-bound, bound] pixels. Flow is returned as a float tensor of shape
    (n_frames, 2, H, W) in units of the quantization bound (i.e., roughly in [-1, 1]); use
    utils.compose_flow_transforms, which rescales the flow when resizing and negates the x flow
    when flipping.

    Args:
        vid_dataset (list): List of (flow_directory_path, label) tuples.
        n_frames (int): Number of flow fields per clip.
        transforms (callable, optional): Flow clip transform applied to a float tensor of shape (T, 2, H, W).
    """
    def load_clip(self, vid_dir):
        """
        Load the flow clip of a video as a float tensor of shape (n_frames, 2, H, W).

        Args:
            vid_dir (str): Path to the video (flow) directory.

        Returns:
            torch.Tensor: The flow clip, in units of the quantization bound.
        """
        x_paths = sorted_frame_paths(vid_dir, prefix='flow_x_')[:self.n_frames]
        y_paths = sorted_frame_paths(vid_dir, prefix='flow_y_')[:self.n_frames]
        if not x_paths or len(x_paths) != len(y_paths):
            raise RuntimeError('Missing flow images in {}'.format(vid_dir))
        flows = [torch.cat([read_image(x_path, mode=ImageReadMode.GRAY), read_image(y_path, mode=ImageReadMode.GRAY)])
                 for x_path, y_path in zip(x_paths, y_paths)]
        flows += [flows[-1]] * (self.n_frames - len(flows))
        return torch.stack(flows).float() / 255 * 2 - 1


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.
    
    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.
    
    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.
    
    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def get_video_group(vid_path):
    """
    Return the source-video group ID of a clip.

    UCF50 clips are named v_<class>_g<group>_c<clip>; all clips sharing a group within a class
    are cut from the same source video (same actor and scene). Clips that do not follow this
    naming scheme are treated as their own group.

    Args:
        vid_path (str): Path to the video (frame) directory.

    Returns:
        str: Group ID of the form <class>_g<group>.
    """
    vid_cat = os.path.basename(os.path.dirname(vid_path))
    vid_name = os.path.basename(vid_path)
    match = re.search(r'_g(\d+)_c\d+$', vid_name)
    return '{}_g{}'.format(vid_cat, match.group(1)) if match else os.path.join(vid_cat, vid_name)


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets using stratified group sampling.

    This function uses StratifiedGroupKFold so that each split has a representative distribution
    of classes, while all clips from the same source video (see get_video_group) are kept in the
    same split. This prevents near-duplicate clips from leaking between train, validation and test.
    Because whole groups are assigned, the resulting ratios are approximate.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    vid_paths = np.array([vid_path for vid_path in vid_dataset.keys()])
    vid_labels = np.array([vid_label for vid_label in vid_dataset.values()])
    vid_groups = np.array([get_video_group(vid_path) for vid_path in vid_paths])
    print('Splitting train/validation/test datasets....')

    # Test split: hold out one of round(1 / ts_ratio) stratified group folds
    ts_spliter = StratifiedGroupKFold(n_splits=round(1 / ts_ratio), shuffle=True, random_state=seed)
    tr_val_idx, ts_idx = next(ts_spliter.split(vid_paths, vid_labels, vid_groups))
    ts_dataset = [(ts_path, ts_label) for ts_path, ts_label in zip(vid_paths[ts_idx], vid_labels[ts_idx])]
    tr_val_paths, tr_val_labels, tr_val_groups = vid_paths[tr_val_idx], vid_labels[tr_val_idx], vid_groups[tr_val_idx]

    # Train/validation split on the remaining groups
    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    val_spliter = StratifiedGroupKFold(n_splits=round(1 / val_wt), shuffle=True, random_state=seed)
    tr_idx, val_idx = next(val_spliter.split(tr_val_paths, tr_val_labels, tr_val_groups))
    tr_dataset = [(tr_path, tr_label) for tr_path, tr_label in zip(tr_val_paths[tr_idx], tr_val_labels[tr_idx])]
    val_dataset = [(val_path, val_label) for val_path, val_label in zip(tr_val_paths[val_idx], tr_val_labels[val_idx])]

    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    
    return padded_imgs, labels_tensor
