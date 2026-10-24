"""
Module: compute_flow.py

This module computes optical flow for the flow stream of the two-stream model. For every video
clip extracted by preprocess.py with --sampling dense (frame0.jpg ... frame{T-1}.jpg), it runs
RAFT-small (torchvision, pretrained on FlyingChairs + FlyingThings) on the T-1 pairs of consecutive
clip frames, so the flow is temporally aligned with the RGB clips used by S3D. The output folder
structure mirrors the input:

    <clip_dir>/<class>/<video>/frame{i}.jpg  ->  <flow_dir>/<class>/<video>/flow_x_{i}.jpg, flow_y_{i}.jpg

flow_x_{i} / flow_y_{i} hold the horizontal / vertical flow from frame i to frame i + 1, clipped to
[-bound, bound] pixels and quantized to [0, 255] grayscale JPEGs (see FlowClipDataset for loading).

Usage:
    python src/compute_flow.py --clip_dir data/UCF50_clips --flow_dir data/UCF50_flow --amp
"""

import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, write_jpeg, ImageReadMode
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from tqdm import tqdm

from video_datasets import sorted_frame_paths


def args_parser():
    """
    Parse command-line arguments for optical flow computation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Compute optical flow for video clips with RAFT')
    parser.add_argument('-cd', '--clip_dir', default='data/UCF50_clips',
                        help='Directory of dense clip frames (preprocess.py --sampling dense)')
    parser.add_argument('-fd', '--flow_dir', default='data/UCF50_flow', help='Directory for storing flow images')
    parser.add_argument('-b', '--bound', type=float, default=20.0,
                        help='Flow values are clipped to [-bound, bound] pixels before quantization')
    parser.add_argument('-nw', '--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes for reading frames')
    parser.add_argument('-p', '--pretrained', default=True, help='Use pretrained RAFT weights')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision on CUDA')
    return parser.parse_args()


class ClipFrames(Dataset):
    """Dataset returning (output_dir, frames) for each video clip, with frames as uint8 (T, 3, H, W)."""
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        clip_path, out_path = self.jobs[idx]
        frames = torch.stack([read_image(p, mode=ImageReadMode.RGB) for p in sorted_frame_paths(clip_path)])
        return out_path, frames


def first_item(batch):
    """Collate function for batch_size=1 that returns the single sample unchanged."""
    return batch[0]


def quantize(flow, bound):
    """Clip flow to [-bound, bound] and map it linearly to uint8 [0, 255]."""
    return ((flow.clamp(-bound, bound) + bound) * (255.0 / (2 * bound))).round().to(torch.uint8)


@torch.no_grad()
def compute_clip_flow(model, frames, device, amp=False):
    """
    Compute RAFT flow between consecutive frames of a clip.

    Args:
        model (nn.Module): RAFT model.
        frames (Tensor): uint8 clip of shape (T, 3, H, W).
        device (torch.device): Device to run RAFT on.
        amp (bool, optional): Use automatic mixed precision. Default is False.

    Returns:
        Tensor: Flow of shape (T - 1, 2, H, W) in pixels (channel 0 = horizontal, 1 = vertical).
    """
    imgs = frames.to(device).float() / 127.5 - 1  # RAFT expects [-1, 1]
    h, w = imgs.shape[-2:]
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8  # RAFT needs sizes divisible by 8
    imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='replicate')
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        flow = model(imgs[:-1], imgs[1:])[-1]
    return flow.float()[..., :h, :w]


def main(args):
    """
    Compute flow for every clip in clip_dir, skipping clips that already have flow.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    jobs = []
    for vid_cat in sorted(os.listdir(args.clip_dir)):
        for vid_name in sorted(os.listdir(os.path.join(args.clip_dir, vid_cat))):
            clip_path = os.path.join(args.clip_dir, vid_cat, vid_name)
            out_path = os.path.join(args.flow_dir, vid_cat, vid_name)
            n_pairs = len(sorted_frame_paths(clip_path)) - 1
            if len(sorted_frame_paths(out_path, prefix='flow_y_')) < n_pairs:
                jobs.append((clip_path, out_path))
    print(f'Computing flow for {len(jobs)} clips....')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = raft_small(weights=Raft_Small_Weights.C_T_V2 if args.pretrained else None).eval().to(device)
    loader = DataLoader(ClipFrames(jobs), batch_size=1, num_workers=args.num_workers, collate_fn=first_item)

    with ThreadPoolExecutor(max_workers=8) as writer:
        for out_path, frames in tqdm(loader):
            flow = quantize(compute_clip_flow(model, frames, device, args.amp), args.bound).cpu()
            os.makedirs(out_path, exist_ok=True)
            for i, (fx, fy) in enumerate(flow):
                writer.submit(write_jpeg, fx.unsqueeze(0), os.path.join(out_path, f'flow_x_{i}.jpg'), quality=95)
                writer.submit(write_jpeg, fy.unsqueeze(0), os.path.join(out_path, f'flow_y_{i}.jpg'), quality=95)


if __name__ == "__main__":
    main(args_parser())
