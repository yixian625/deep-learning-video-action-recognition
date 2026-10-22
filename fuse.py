"""
Module: fuse.py

This module performs late fusion of a two-stream (or multi-stream) model. Each stream is evaluated
with run.py --mode eval, which writes test_predictions.npz (video_keys, labels, probs) next to its
checkpoint. This module matches the predictions per video key, averages the class probabilities
of the streams (optionally weighted), and reports the test accuracy of each stream and of the
fused model, logging them to Weights & Biases.

The fusion weights must not be tuned on the test set. By default the streams are averaged with
equal weights; with --val_preds (the streams' val_predictions.npz from run.py --eval_split val),
the weights of two streams are chosen by a grid search on the validation set and then applied to
the test set.

Usage:
    python fuse.py --preds models/s3d/test_predictions.npz models/s3d_flow/test_predictions.npz --names rgb flow
    python fuse.py --preds <test npz files> --val_preds <val npz files> --names rgb flow
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
    parser.add_argument('--val_preds', nargs='+',
                        help='val_predictions.npz file of each stream; if given, the fusion weights of two streams '
                             'are chosen on the validation set (overrides --weights)')
    parser.add_argument('--grid_step', type=float, default=0.05, help='Step of the weight grid search (default 0.05)')
    parser.add_argument('-rn', '--run_name', default='fusion', help="W&B run name (default 'fusion')")
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


def choose_weights_on_val(val_pred_paths, grid_step=0.05):
    """
    Choose the fusion weights of two streams by a grid search on the validation set.

    The weight of the second stream w runs over [0, 1] in steps of grid_step (the first stream gets
    1 - w). The weight with the highest fused validation accuracy is chosen; ties are broken by the
    lower fused validation log loss.

    Args:
        val_pred_paths (list): Paths to the two streams' val_predictions.npz files.
        grid_step (float, optional): Step of the grid. Default is 0.05.

    Returns:
        tuple: (weights, val_results)
            - weights (np.ndarray): Chosen weights of the two streams.
            - val_results (dict): Per-stream and fused validation accuracy at the chosen weights.
    """
    if len(val_pred_paths) != 2:
        raise ValueError('Choosing weights on the validation set is supported for two streams only.')
    _, labels, probs = load_aligned(val_pred_paths)
    best = None
    for w in np.round(np.arange(0, 1 + 1e-9, grid_step), 6):
        fused = (1 - w) * probs[0] + w * probs[1]
        acc = (fused.argmax(1) == labels).mean()
        log_loss = -np.log(fused[np.arange(len(labels)), labels] + 1e-12).mean()
        if best is None or (acc, -log_loss) > (best[1], -best[2]):
            best = (w, acc, log_loss)
    w, acc, _ = best
    val_results = {'val/stream{}_accuracy'.format(i): (p.argmax(1) == labels).mean() for i, p in enumerate(probs)}
    val_results['val/fused_accuracy'] = acc
    return np.array([1 - w, w]), val_results


def main(args):
    """
    Fuse the streams' predictions, print and log per-stream and fused test accuracy.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    names = args.names or ['stream{}'.format(i) for i in range(len(args.preds))]
    val_results = {}
    if args.val_preds:
        weights, val_results = choose_weights_on_val(args.val_preds, args.grid_step)
        val_results = {k.replace('stream0', names[0]).replace('stream1', names[1]): v for k, v in val_results.items()}
        print('Weights chosen on the validation set: {}'.format(dict(zip(names, np.round(weights, 3)))))
    else:
        weights = np.array(args.weights or [1.0] * len(args.preds))
    weights = weights / weights.sum()

    video_keys, labels, probs = load_aligned(args.preds)
    fused = sum(w * p for w, p in zip(weights, probs))

    results = {'test/{}_accuracy'.format(name): (p.argmax(1) == labels).mean() for name, p in zip(names, probs)}
    results['test/fused_accuracy'] = (fused.argmax(1) == labels).mean()
    results.update(val_results)
    for name, acc in results.items():
        print('{}: {:.2f}%'.format(name, 100 * acc))

    wandb.init(project='video-action-lrcn', job_type='fusion', mode=args.wandb_mode, name=args.run_name,
               tags=['fusion'] + names,
               config={'preds': args.preds, 'val_preds': args.val_preds, 'names': names, 'weights': weights.tolist(),
                       'n_videos': len(video_keys)})
    wandb.log(results)
    for key, value in results.items():
        wandb.run.summary[key] = value
    wandb.finish()


if __name__ == "__main__":
    args = args_parser()
    main(args)
