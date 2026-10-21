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
    
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import wandb

from video_datasets import VideoDataset, ClipDataset, FlowClipDataset, load_dataset, dataset_split
from utils import (transform_stats, compose_data_transforms, compose_clip_transforms, compose_flow_transforms,
                   train_val_dloaders, test_dloaders)
from models import build_model, get_param_groups
from train import train
from test import test, get_test_report, get_confusion_matrix

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
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-trs', '--train_size', type=float, default=0.8, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', default='lrcn', choices=['lrcn', 's3d', 's3d_flow'],
                        help='LRCN (2D CNN + LSTM), S3D (3D CNN) on RGB, or S3D on optical flow')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', help='Use pretrained weights (ImageNet for LRCN, Kinetics-400 for S3D)', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
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
                        help='Stop after this many consecutive epochs without a decrease in validation loss (default: off)')
    parser.add_argument('-wm', '--wandb_mode', default='online', choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases logging mode')
    parser.add_argument('-od', '--output_dir', help="Directory for model weights and splits (default './models/<model_type>')")
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of DataLoader worker processes')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision on CUDA')
    parser.add_argument('-rn', '--run_name', help="W&B run name (default '<model_type>-<mode>')")

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

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN or S3D) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set, prints the overall test accuracy, and saves
          the per-video class probabilities next to the checkpoint.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    num_workers = args.num_workers
    amp = args.amp
    output_dir = args.output_dir or os.path.join('./models', model_type)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Start a Weights & Biases run to track this experiment
    wandb.init(project='video-action-lrcn', job_type=mode, config=vars(args), mode=args.wandb_mode,
               name=args.run_name or '{}-{}'.format(model_type, mode), tags=[model_type])

    # Load transformation statistics and create data augmentation transforms
    # (video-level for LRCN, clip-level for S3D, flow-aware for the flow stream)
    h, w, mean, std = transform_stats(model_type)
    compose_transforms = {'lrcn': compose_data_transforms, 's3d': compose_clip_transforms,
                          's3d_flow': compose_flow_transforms}[model_type]
    tr_transforms, val_ts_transforms = compose_transforms(h, w, mean, std)

    # Initialize the model (LRCN or S3D)
    model = build_model(model_type, n_classes, pretrained=pretrained, dropout_rate=dropout,
                        rnn_hidden_size=rnn_hidden_size, rnn_n_layers=rnn_n_layers, cnn_backbone=cnn_backbone)

    if mode == 'train':
        # Load dataset and split into train/validation/test
        vid_dataset, label_dict = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

        # Save the splits next to the model weights for reproducibility and later use in evaluation
        os.makedirs(output_dir, exist_ok=True)
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        np.save(os.path.join(output_dir, 'splits.npy'), splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset, val_dataset = build_datasets(model_type, [tr_split, val_split], fr_per_vid,
                                                 [tr_transforms, val_ts_transforms])
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type, num_workers)

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(get_param_groups(model, model_type, learning_rate, args.backbone_lr))
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        # Main training procedure
        model.to(device)
        model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device, output_dir, n_epochs, amp,
                                           args.early_stopping_patience)

    elif mode == 'eval':
        # Load the dataset splits saved next to the checkpoint during training
        ckpt_dir = os.path.dirname(ckpt)
        splits = np.load(os.path.join(ckpt_dir, 'splits.npy'), allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset, = build_datasets(model_type, [ts_split], fr_per_vid, [val_ts_transforms])
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type, num_workers)

        # Load the trained model checkpoint
        model.load_state_dict(torch.load(ckpt))
        model.to(device)
        targets, outputs, accuracy, probs = test(model, dataloaders['test'], device, amp)

        print('The overall test accuracy is {:.4f}%.'.format(100 * accuracy))

        # Save per-video class probabilities; streams evaluated on the same split can be fused
        # by averaging their probabilities per video key
        np.savez(os.path.join(ckpt_dir, 'test_predictions.npz'),
                 video_keys=np.array([video_key(vid_path) for vid_path, _ in ts_split]),
                 labels=np.array(targets), probs=probs)

        # Recover class names from the saved split paths (<frame_dir>/<class>/<video>)
        labels_dict = {}
        for split in splits.item().values():
            for vid_path, label in split:
                labels_dict[os.path.basename(os.path.dirname(vid_path))] = int(label)
        all_cats = sorted(labels_dict, key=labels_dict.get)

        # Log test accuracy, per-class report and confusion matrix to W&B
        report = get_test_report(targets, outputs, all_cats)
        report_table = wandb.Table(columns=['class', 'precision', 'recall', 'f1-score', 'support'])
        for cat in all_cats:
            report_table.add_data(cat, *[report[cat][k] for k in ['precision', 'recall', 'f1-score', 'support']])
        wandb.log({
            'test/accuracy': accuracy,
            'test/classification_report': report_table,
            'test/confusion_matrix': wandb.plot.confusion_matrix(
                y_true=targets, preds=outputs, class_names=all_cats),
        })
        wandb.run.summary['test/accuracy'] = accuracy
        wandb.run.summary['test/macro_f1'] = report['macro avg']['f1-score']

    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

    wandb.finish()

if __name__ == "__main__":
    args = args_parser()
    main(args)
