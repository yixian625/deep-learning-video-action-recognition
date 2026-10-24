"""
Regression tests for frame ordering (README "Fixes" #2).

Frames are stored as frame0.jpg ... frame16.jpg. VideoDataset must load them in numeric order;
the original implementation used unsorted glob, which yields string order on most file systems
(frame0, frame1, frame10, frame11, ..., frame2, ...).
"""

import numpy as np
import pytest
from PIL import Image
import torch
from torchvision.transforms import v2

from video_datasets import VideoDataset

N_STORED_FRAMES = 17  # get_frames(n_frames=16) stores 17 frames
STEP = 10  # frame i is a solid gray image with intensity i * STEP


@pytest.fixture(name='video_dir')
def fixture_video_dir(tmp_path):
    """Create a video directory whose frame i encodes i in its pixel values."""
    vid_dir = tmp_path / 'Biking' / 'v_Biking_g01_c01'
    vid_dir.mkdir(parents=True)
    for i in range(N_STORED_FRAMES):
        img = np.full((8, 8, 3), i * STEP, dtype=np.uint8)
        Image.fromarray(img).save(vid_dir / f'frame{i}.jpg')
    return str(vid_dir)


def loaded_frame_indices(video_dir, fr_per_vid):
    """Load a video with VideoDataset and return the frame indices it loaded, in order, and its label."""
    dataset = VideoDataset([(video_dir, 0)], fr_per_vid, v2.ToDtype(torch.float32, scale=True))
    frames, label = dataset[0]
    # Decode each frame's index from its mean intensity (JPEG is lossy, so round)
    return [int(round(frame.mean().item() * 255 / STEP)) for frame in frames], label


def test_frames_are_loaded_in_numeric_order(video_dir):
    """All 17 frames are loaded in the order frame0, frame1, ..., frame16."""
    indices, _ = loaded_frame_indices(video_dir, fr_per_vid=N_STORED_FRAMES)
    assert indices == list(range(N_STORED_FRAMES))


def test_fr_per_vid_keeps_the_first_frames_in_time(video_dir):
    """fr_per_vid=16 keeps frames 0-15, not the first 16 in string order."""
    # With string ordering, the first 16 would include frame10..frame16 and drop frame8/frame9
    indices, _ = loaded_frame_indices(video_dir, fr_per_vid=16)
    assert indices == list(range(16))


def test_frame_tensor_shape_and_label(video_dir):
    """The loaded frames are stacked into one (T, C, H, W) tensor and the label is returned unchanged."""
    dataset = VideoDataset([(video_dir, 7)], 16, v2.ToDtype(torch.float32, scale=True))
    frames, label = dataset[0]
    assert tuple(frames.shape) == (16, 3, 8, 8)
    assert label == 7
