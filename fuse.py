"""
Module: fuse.py

This module performs late fusion of a two-stream (or multi-stream) model. Each stream is evaluated
with run.py --mode eval, which writes test_predictions.npz (video_keys, labels, probs) next to its
checkpoint. This module matches the predictions per video key, averages the class probabilities
of the streams (optionally weighted), and reports the test accuracy of each stream and of the
fused model, logging them to Weights & Biases.

The fusion weights should not be tuned on the test set; the default is an equal-weight average.

Usage:
    python fuse.py --preds models/s3d/test_predictions.npz models/s3d_flow/test_predictions.npz --names rgb flow
"""

import argparse

import numpy as np
import wandb


def args_parser():
    """
    Parse command-line arguments for late fusion.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Late fusion of per-stream test predictions')
    parser.add_argument('--preds', nargs='+', required=True, help='test_predictions.npz file of each stream')
    parser.add_argument('--names', nargs='+', help='Name of each stream (default: stream0, stream1, ...)')
    parser.add_argument('--weights', nargs='+', type=float, help='Fusion weight of each stream (default: equal)')
    parser.add_argument('-wm', '--wandb_mode', default='online', choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases logging mode')
    return parser.parse_args()


def load_aligned(pred_paths):
    """
    Load per-stream predictions and align them by video key.

    Args:
        pred_paths (list): Paths to test_predictions.npz files.

    Returns:
        tuple: (video_keys, labels, probs)
            - video_keys (np.ndarray): Sorted video keys shared by all streams.
            - labels (np.ndarray): Ground truth label of each video.
            - probs (list): Per-stream class probabilities of shape (n_videos, n_classes), in video_keys order.

    Raises:
        ValueError: If the streams were evaluated on different videos or disagree on labels.
    """
    preds = [np.load(path) for path in pred_paths]
    video_keys = np.sort(preds[0]['video_keys'])
    labels, probs = None, []
    for path, pred in zip(pred_paths, preds):
        if set(pred['video_keys']) != set(video_keys):
            raise ValueError('{} was evaluated on different videos than {}'.format(path, pred_paths[0]))
        order = np.argsort(pred['video_keys'])
        if labels is None:
            labels = pred['labels'][order]
        elif not np.array_equal(labels, pred['labels'][order]):
            raise ValueError('{} has different labels than {}'.format(path, pred_paths[0]))
        probs.append(pred['probs'][order])
    return video_keys, labels, probs


def main(args):
    """
    Fuse the streams' predictions, print and log per-stream and fused test accuracy.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    names = args.names or ['stream{}'.format(i) for i in range(len(args.preds))]
    weights = np.array(args.weights or [1.0] * len(args.preds))
    weights = weights / weights.sum()

    video_keys, labels, probs = load_aligned(args.preds)
    fused = sum(w * p for w, p in zip(weights, probs))

    results = {'test/{}_accuracy'.format(name): (p.argmax(1) == labels).mean() for name, p in zip(names, probs)}
    results['test/fused_accuracy'] = (fused.argmax(1) == labels).mean()
    for name, acc in results.items():
        print('{}: {:.2f}%'.format(name, 100 * acc))

    wandb.init(project='video-action-lrcn', job_type='fusion', mode=args.wandb_mode, tags=['fusion'] + names,
               config={'preds': args.preds, 'names': names, 'weights': weights.tolist(), 'n_videos': len(video_keys)})
    wandb.log(results)
    for key, value in results.items():
        wandb.run.summary[key] = value
    wandb.finish()


if __name__ == "__main__":
    args = args_parser()
    main(args)
