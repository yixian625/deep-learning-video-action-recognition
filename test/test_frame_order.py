"""
Regression tests for frame ordering (README "Fixes" #2).

Frames are stored as frame0.jpg ... frame16.jpg. VideoDataset must load them in numeric order;
the original implementation used unsorted glob, which yields string order on most file systems
(frame0, frame1, frame10, frame11, ..., frame2, ...).
"""

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from video_datasets import VideoDataset

N_STORED_FRAMES = 17  # get_frames(n_frames=16) stores 17 frames
STEP = 10  # frame i is a solid gray image with intensity i * STEP


@pytest.fixture
def video_dir(tmp_path):
    """Create a video directory whose frame i encodes i in its pixel values."""
    vid_dir = tmp_path / 'Biking' / 'v_Biking_g01_c01'
    vid_dir.mkdir(parents=True)
    for i in range(N_STORED_FRAMES):
        img = np.full((8, 8, 3), i * STEP, dtype=np.uint8)
        Image.fromarray(img).save(vid_dir / 'frame{}.jpg'.format(i))
    return str(vid_dir)


def loaded_frame_indices(video_dir, fr_per_vid):
    dataset = VideoDataset([(video_dir, 0)], fr_per_vid, transforms.ToTensor())
    frames, label = dataset[0]
    # Decode each frame's index from its mean intensity (JPEG is lossy, so round)
    return [int(round(frame.mean().item() * 255 / STEP)) for frame in frames], label


def test_frames_are_loaded_in_numeric_order(video_dir):
    indices, _ = loaded_frame_indices(video_dir, fr_per_vid=N_STORED_FRAMES)
    assert indices == list(range(N_STORED_FRAMES))


def test_fr_per_vid_keeps_the_first_frames_in_time(video_dir):
    # With string ordering, the first 16 would include frame10..frame16 and drop frame8/frame9
    indices, _ = loaded_frame_indices(video_dir, fr_per_vid=16)
    assert indices == list(range(16))


def test_frame_tensor_shape_and_label(video_dir):
    dataset = VideoDataset([(video_dir, 7)], 16, transforms.ToTensor())
    frames, label = dataset[0]
    assert tuple(frames.shape) == (16, 3, 8, 8)
    assert label == 7
