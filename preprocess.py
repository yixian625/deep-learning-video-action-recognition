"""
Module: preprocess.py

This module converts a video dataset (e.g., UCF50) organized as one subdirectory of video
files per action class into folders of extracted frames, which is the format expected by
VideoDataset. The output folder structure mirrors the input:

    <video_dir>/<class>/<video>.avi  ->  <frame_dir>/<class>/<video>/frame{i}.jpg

Usage:
    python preprocess.py -vd data/UCF50 -fd data/UCF50_frames -nf 16
"""

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from utils import get_frames, store_frames


def args_parser():
    """
    Parse command-line arguments for frame extraction.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('-vd', '--video_dir', default='data/UCF50', help='Directory of class subdirectories containing videos')
    parser.add_argument('-fd', '--frame_dir', default='data/UCF50_frames', help='Directory for storing extracted frames')
    parser.add_argument('-nf', '--n_frames', type=int, default=16, help='Number of frames to sample per video')
    parser.add_argument('-nw', '--n_workers', type=int, default=os.cpu_count(), help='Number of parallel worker processes')
    return parser.parse_args()


def extract_video(vid_path, store_path, n_frames):
    """
    Extract frames from a single video and store them as JPEG images.

    Args:
        vid_path (str): Path to the video file.
        store_path (str): Directory where the frames will be stored.
        n_frames (int): Number of frames to sample from the video.
    """
    os.makedirs(store_path, exist_ok=True)
    frames, _ = get_frames(vid_path, n_frames=n_frames)
    store_frames(frames, store_path)


def main(args):
    """
    Extract frames for every video in video_dir, skipping videos that already have frames.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    jobs = []
    for vid_cat in sorted(os.listdir(args.video_dir)):
        for vid_path in sorted(glob.glob(os.path.join(args.video_dir, vid_cat, '*.avi'))):
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            store_path = os.path.join(args.frame_dir, vid_cat, vid_name)
            if not glob.glob(os.path.join(store_path, '*.jpg')):
                jobs.append((vid_path, store_path))

    print('Extracting frames from {} videos....'.format(len(jobs)))
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(extract_video, vid_path, store_path, args.n_frames)
                   for vid_path, store_path in jobs]
        for future in tqdm(futures):
            future.result()


if __name__ == "__main__":
    args = args_parser()
    main(args)
