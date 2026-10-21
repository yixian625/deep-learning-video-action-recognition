"""
Regression test for per-frame augmentation (README "Fixes" #4).

The original LRCN pipeline applied RandomHorizontalFlip and RandomAffine to each frame separately,
so frames of the same video were flipped and shifted differently, adding fake motion.
Random augmentations must be identical for all frames of a video.
"""

import numpy as np
import torch
from PIL import Image

from utils import compose_data_transforms, transform_stats
from video_datasets import VideoDataset


def test_random_augmentation_is_identical_for_all_frames_of_a_video(tmp_path):
    # Identical frames with a horizontal gradient, so a flip or shift changes the frame
    vid_dir = tmp_path / 'Class0' / 'v_Class0_g01_c01'
    vid_dir.mkdir(parents=True)
    img = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
    for i in range(8):
        Image.fromarray(np.stack([img] * 3, axis=-1)).save(vid_dir / 'frame{}.jpg'.format(i))

    tr_transforms, _ = compose_data_transforms(*transform_stats('lrcn'))
    dataset = VideoDataset([(str(vid_dir), 0)], 8, tr_transforms)
    for seed in range(10):
        torch.manual_seed(seed)
        frames, _ = dataset[0]
        for frame in frames[1:]:
            torch.testing.assert_close(frame, frames[0])
