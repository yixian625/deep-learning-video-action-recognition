"""
Module: preprocess.py

This module converts a video dataset (e.g., UCF50) organized as one subdirectory of video
files per action class into folders of extracted frames, which is the format expected by
VideoDataset and ClipDataset. The output folder structure mirrors the input:

    <video_dir>/<class>/<video>.avi  ->  <frame_dir>/<class>/<video>/frame{i}.jpg

Two sampling modes are supported:
    - uniform: frames spread evenly over the whole video (for LRCN), see utils.get_frames.
    - dense: a clip of consecutive frames taken every `stride` frames from the middle of the
      video (for 3D CNNs such as S3D, and as the source frames for optical flow), see
      utils.get_clip_frames.

Usage:
    python src/preprocess.py -vd data/UCF50 -fd data/UCF50_frames -nf 16
    python src/preprocess.py -vd data/UCF50 -fd data/UCF50_clips -nf 32 --sampling dense --stride 2
"""

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from utils import get_frames, get_clip_frames, store_frames


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
    parser.add_argument('-s', '--sampling', default='uniform', choices=['uniform', 'dense'],
                        help="'uniform': spread over the whole video (LRCN); 'dense': consecutive strided clip (3D CNN)")
    parser.add_argument('--stride', type=int, default=2, help='Temporal stride between clip frames (dense sampling only)')
    return parser.parse_args()


def extract_video(vid_path, store_path, n_frames, sampling='uniform', stride=2):
    """
    Extract frames from a single video and store them as JPEG images.

    Args:
        vid_path (str): Path to the video file.
        store_path (str): Directory where the frames will be stored.
        n_frames (int): Number of frames to sample from the video.
        sampling (str, optional): 'uniform' or 'dense'. Default is 'uniform'.
        stride (int, optional): Temporal stride between clip frames for dense sampling. Default is 2.
    """
    os.makedirs(store_path, exist_ok=True)
    if sampling == 'dense':
        frames, _ = get_clip_frames(vid_path, n_frames=n_frames, stride=stride)
    else:
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
        futures = [executor.submit(extract_video, vid_path, store_path, args.n_frames, args.sampling, args.stride)
                   for vid_path, store_path in jobs]
        for future in tqdm(futures):
            future.result()


if __name__ == "__main__":
    args = args_parser()
    main(args)
