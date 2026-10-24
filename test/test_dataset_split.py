"""
Regression tests for the train/val/test data leak (README "Fixes" #1).

UCF50 clips named v_<class>_g<group>_c<clip> that share a group are cut from the same source
video. dataset_split must keep each group entirely within one split; the original clip-level
StratifiedShuffleSplit spread clips of a group across train/val/test.
"""

import os

import pytest

from video_datasets import dataset_split, get_video_group


@pytest.fixture(name='splits', scope='module')
def fixture_splits():
    """Split a UCF50-style dataset of 5 classes x 25 groups x 4 clips (no files needed)."""
    vid_dataset = {}
    for label in range(5):
        cat = f'Class{label}'
        for g in range(1, 26):
            for c in range(1, 5):
                vid_dataset[os.path.join('frames', cat, f'v_{cat}_g{g:02d}_c{c:02d}')] = label
    return dataset_split(vid_dataset, tr_ratio=0.8, ts_ratio=0.1, seed=0)


def test_clips_of_the_same_source_video_share_a_group():
    """All clips v_<class>_g07_c* of one source video get the same group ID."""
    paths = [os.path.join('frames', 'Biking', f'v_Biking_g07_c{c:02d}') for c in range(1, 6)]
    assert {get_video_group(p) for p in paths} == {'Biking_g07'}


def test_no_source_video_group_is_shared_between_splits(splits):
    """No source-video group appears in more than one of train, val and test."""
    tr_g, val_g, ts_g = ({get_video_group(path) for path, _ in split} for split in splits)
    assert not tr_g & val_g, 'groups leaked between train and val'
    assert not tr_g & ts_g, 'groups leaked between train and test'
    assert not val_g & ts_g, 'groups leaked between val and test'
