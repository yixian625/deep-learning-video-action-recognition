"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (LRCN or S3D) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights and the splits to the output directory.

- Eval mode:
    - Loads the dataset splits saved next to the checkpoint.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set, prints overall test accuracy, and saves the per-video
      class probabilities (test_predictions.npz) next to the checkpoint, e.g. for two-stream late fusion.
      With --eval_split val, it evaluates the validation set instead (val_predictions.npz), e.g. to
      choose fusion weights without using the test set.

The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import wandb

from video_datasets import VideoDataset, ClipDataset, FlowClipDataset, load_dataset, dataset_split
from utils import (transform_stats, compose_data_transforms, compose_clip_transforms, compose_flow_transforms,
                   train_val_dloaders, test_dloaders)
from models import build_model, get_param_groups
from train import train, TrainConfig
from evaluate import test, get_test_report

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.8).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video (LRCN) or per clip (S3D) to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type: 'lrcn', 's3d' (RGB) or 's3d_flow' (optical flow) (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use pretrained weights: ImageNet for LRCN, Kinetics-400 for S3D (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training; of the head only if --backbone_lr is set (default 3e-5).
        -blr/--backbone_lr: Learning rate of the pretrained backbone (default: same as --learning_rate).
        -ne/--n_epochs: Maximum number of training epochs (default 30).
        -es/--early_stopping_patience: Stop after this many epochs without a decrease in val loss (default off).
        -wm/--wandb_mode: Weights & Biases mode: 'online', 'offline' or 'disabled' (default 'online').
        -od/--output_dir: Directory for the best model weights and splits (default './models/<model_type>').
        -nw/--num_workers: Number of DataLoader worker processes (default 0).
        --amp: Use automatic mixed precision on CUDA (default off).
        -rn/--run_name: W&B run name (default '<model_type>-<mode>').
        -bnm/--bn_momentum: S3D only: BatchNorm momentum (default: torchvision's 0.001).
        -esp/--eval_split: Split evaluated in eval mode: 'test' or 'val' (default 'test').
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-trs', '--train_size', type=float, default=0.8, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int,
                        help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', default='lrcn', choices=['lrcn', 's3d', 's3d_flow'],
                        help='LRCN (2D CNN + LSTM), S3D (3D CNN) on RGB, or S3D on optical flow')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                        help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained',
                        help='Use pretrained weights (ImageNet for LRCN, Kinetics-400 for S3D)', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100,
                        help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5,
                        help='Learning rate for model training (of the head only if --backbone_lr is set)')
    parser.add_argument('-blr', '--backbone_lr', type=float,
                        help='Learning rate of the pretrained backbone (default: same as --learning_rate)')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Maximum number of training epochs')
    parser.add_argument('-es', '--early_stopping_patience', type=int,
                        help='Stop after this many consecutive epochs without a decrease in validation loss '
                             '(default: off)')
    parser.add_argument('-wm', '--wandb_mode', default='online', choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases logging mode')
    parser.add_argument('-od', '--output_dir',
                        help="Directory for model weights and splits (default './models/<model_type>')")
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of DataLoader worker processes')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision on CUDA')
    parser.add_argument('-rn', '--run_name', help="W&B run name (default '<model_type>-<mode>')")
    parser.add_argument('-esp', '--eval_split', default='test', choices=['test', 'val'],
                        help="Split evaluated in eval mode (default 'test'; 'val' e.g. for choosing fusion weights)")
    parser.add_argument('-bnm', '--bn_momentum', type=float,
                        help="S3D only: momentum of the BatchNorm running statistics (default: torchvision's 0.001)")

    args = parser.parse_args()
    return args

def build_datasets(model_type, splits, fr_per_vid, transforms):
    """
    Create a PyTorch Dataset for each split, using the dataset class that matches the model type.

    LRCN uses VideoDataset, S3D uses ClipDataset (RGB clips) and the flow stream uses FlowClipDataset.

    Args:
        model_type (str): 'lrcn', 's3d' or 's3d_flow'.
        splits (list): List of splits, each a list of (video_path, label) tuples.
        fr_per_vid (int): Number of frames per video / clip.
        transforms (list): Transform for each split.

    Returns:
        list: One Dataset per split.
    """
    dataset_cls = {'lrcn': VideoDataset, 's3d': ClipDataset, 's3d_flow': FlowClipDataset}[model_type]
    return [dataset_cls(split, fr_per_vid, trans) for split, trans in zip(splits, transforms)]


def video_key(vid_path):
    """Return a frame-directory-independent key '<class>/<video>' for a video path."""
    return '/'.join(os.path.normpath(str(vid_path)).split(os.sep)[-2:])


def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    Starts a Weights & Biases run, builds the data transforms and the model, and then either
    trains the model (run_train) or evaluates a trained checkpoint (run_eval).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Start a Weights & Biases run to track this experiment
    wandb.init(project='video-action-lrcn', job_type=args.mode, config=vars(args), mode=args.wandb_mode,
               name=args.run_name or f'{args.model_type}-{args.mode}', tags=[args.model_type])

    # Load transformation statistics and create data augmentation transforms
    # (video-level for LRCN, clip-level for S3D, flow-aware for the flow stream)
    h, w, mean, std = transform_stats(args.model_type)
    compose_transforms = {'lrcn': compose_data_transforms, 's3d': compose_clip_transforms,
                          's3d_flow': compose_flow_transforms}[args.model_type]
    tr_transforms, val_ts_transforms = compose_transforms(h, w, mean, std)

    # Initialize the model (LRCN or S3D)
    model = build_model(args.model_type, args.n_classes, pretrained=args.pretrained, dropout_rate=args.dropout,
                        rnn_hidden_size=args.rnn_hidden_size, rnn_n_layers=args.rnn_n_layers,
                        cnn_backbone=args.cnn_backbone, bn_momentum=args.bn_momentum)

    if args.mode == 'train':
        run_train(args, model, (tr_transforms, val_ts_transforms), device)
    elif args.mode == 'eval':
        run_eval(args, model, val_ts_transforms, device)
    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

    wandb.finish()


def run_train(args, model, transforms, device):
    """
    Split the dataset, save the splits, and train the model.

    The splits and the best model weights are saved to --output_dir (default './models/<model_type>').

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The model to train.
        transforms (tuple): (train_transforms, val_transforms).
        device (torch.device): Device (CPU or GPU) on which to train.
    """
    output_dir = args.output_dir or os.path.join('./models', args.model_type)

    # Load dataset and split into train/validation/test
    vid_dataset, _ = load_dataset(args.frame_dir)
    tr_split, val_split, ts_split = dataset_split(vid_dataset, args.train_size, args.test_size)

    # Save the splits next to the model weights for reproducibility and later use in evaluation
    os.makedirs(output_dir, exist_ok=True)
    splits = {'train': np.array(tr_split),
              'val': np.array(val_split),
              'test': np.array(ts_split)}
    np.save(os.path.join(output_dir, 'splits.npy'), splits)

    # Create PyTorch Datasets and DataLoaders for train and validation
    tr_dataset, val_dataset = build_datasets(args.model_type, [tr_split, val_split], args.fr_per_vid, list(transforms))
    dataloaders = train_val_dloaders(tr_dataset, val_dataset, args.batch_size, args.model_type, args.num_workers)

    # Define the loss function, optimizer, and learning rate scheduler
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(get_param_groups(model, args.model_type, args.learning_rate, args.backbone_lr))
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    # Main training procedure
    model.to(device)
    train(dataloaders, model, loss_func, opt, lr_scheduler, device,
          TrainConfig(output_dir, args.n_epochs, args.amp, args.early_stopping_patience))


def run_eval(args, model, transform, device):
    """
    Evaluate a trained checkpoint on the test (or validation) split saved next to it.

    Prints the accuracy, saves the per-video class probabilities ('<split>_predictions.npz') next to
    the checkpoint (e.g. for two-stream late fusion), and logs the results to W&B.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The model to load the checkpoint into.
        transform (callable): Validation/test transform.
        device (torch.device): Device (CPU or GPU) on which to evaluate.
    """
    # Load the dataset splits saved next to the checkpoint during training
    ckpt_dir = os.path.dirname(args.ckpt)
    splits = np.load(os.path.join(ckpt_dir, 'splits.npy'), allow_pickle=True).item()
    split_name = args.eval_split
    eval_split = [(sample[0], int(sample[1])) for sample in splits[split_name]]

    # Create PyTorch Dataset and DataLoader for the evaluated split
    eval_dataset, = build_datasets(args.model_type, [eval_split], args.fr_per_vid, [transform])
    dataloader = test_dloaders(eval_dataset, args.batch_size, args.model_type, args.num_workers)['test']

    # Load the trained model checkpoint
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    targets, outputs, accuracy, probs = test(model, dataloader, device, args.amp)

    print(f'The overall {split_name} accuracy is {100 * accuracy:.4f}%.')

    # Save per-video class probabilities; streams evaluated on the same split can be fused
    # by averaging their probabilities per video key
    np.savez(os.path.join(ckpt_dir, f'{split_name}_predictions.npz'),
             video_keys=np.array([video_key(vid_path) for vid_path, _ in eval_split]),
             labels=np.array(targets), probs=probs)

    log_eval_results(split_name, targets, outputs, accuracy, class_names(splits))


def class_names(splits):
    """
    Recover the class names, ordered by label, from the saved split paths (<frame_dir>/<class>/<video>).

    Args:
        splits (dict): Splits as saved in splits.npy ({'train': ..., 'val': ..., 'test': ...}).

    Returns:
        list: Class names, where the i-th name belongs to label i.
    """
    labels_dict = {}
    for split in splits.values():
        for vid_path, label in split:
            labels_dict[os.path.basename(os.path.dirname(vid_path))] = int(label)
    return sorted(labels_dict, key=labels_dict.get)


def log_eval_results(split_name, targets, outputs, accuracy, all_cats):
    """
    Log the accuracy, per-class report and confusion matrix of an evaluated split to W&B.

    Args:
        split_name (str): 'test' or 'val' (used as the W&B key prefix).
        targets (list): Ground truth labels.
        outputs (list): Predicted labels.
        accuracy (float): Overall accuracy.
        all_cats (list): Class names, ordered by label.
    """
    report = get_test_report(targets, outputs, all_cats)
    report_table = wandb.Table(columns=['class', 'precision', 'recall', 'f1-score', 'support'])
    for cat in all_cats:
        report_table.add_data(cat, *[report[cat][k] for k in ['precision', 'recall', 'f1-score', 'support']])
    wandb.log({
        f'{split_name}/accuracy': accuracy,
        f'{split_name}/classification_report': report_table,
        f'{split_name}/confusion_matrix': wandb.plot.confusion_matrix(
            y_true=targets, preds=outputs, class_names=all_cats),
    })
    wandb.run.summary[f'{split_name}/accuracy'] = accuracy
    wandb.run.summary[f'{split_name}/macro_f1'] = report['macro avg']['f1-score']


if __name__ == "__main__":
    main(args_parser())
