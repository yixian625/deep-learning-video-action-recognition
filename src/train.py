"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, optionally stops early
when the validation loss stops decreasing, and provides helper functions to compute the
learning rate, batch loss, and epoch loss.

Classes:
    - TrainConfig: Options of a training run (output folder, epochs, mixed precision, early stopping).
    - BestModelTracker: Keeps and saves the weights with the best validation accuracy.
    - EarlyStopping: Counts epochs without a decrease in validation loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - log_epoch: Logs the metrics of an epoch to the console and W&B.
    - get_learning_rates: Retrieves the current learning rate of every named parameter group.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm
import torch
import wandb


@dataclass
class TrainConfig:
    """
    Options of a training run.

    Attributes:
        output_dir (str): Directory to save the best model weights.
        n_epochs (int): Maximum number of training epochs. Default is 30.
        amp (bool): Use automatic mixed precision (float16) on CUDA. Default is False.
        early_stopping_patience (int, optional): Stop after this many consecutive epochs without a
            decrease in validation loss. Default is None (always train for n_epochs).
    """
    output_dir: str
    n_epochs: int = 30
    amp: bool = False
    early_stopping_patience: Optional[int] = None


class BestModelTracker:
    """
    Keeps the model weights with the best validation accuracy and saves them to disk.

    Args:
        model (torch.nn.Module): The model being trained (its initial weights are the first "best").
        output_dir (str): Directory to save the best model weights to (as best_model_wts.pt).
    """
    def __init__(self, model, output_dir):
        self.best_accuracy = 0.0
        self.best_weights = copy.deepcopy(model.state_dict())
        self.path = os.path.join(output_dir, 'best_model_wts.pt')

    def update(self, model, val_accuracy, epoch):
        """Save the model weights if val_accuracy is higher than the best so far."""
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_weights = copy.deepcopy(model.state_dict())
            torch.save(self.best_weights, self.path)
            print(f'Best model weights are updated at epoch {epoch}!')

    def restore(self, model):
        """Load the best weights so far into the model."""
        model.load_state_dict(self.best_weights)


class EarlyStopping:
    """
    Counts consecutive epochs without a decrease in validation loss.

    Args:
        patience (int, optional): Number of such epochs after which training should stop.
            Default is None (never stop early; the best validation loss is still tracked).
    """
    def __init__(self, patience=None):
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_since_best = 0

    def update(self, val_loss):
        """Record the validation loss of an epoch."""
        if val_loss < self.best_loss:
            self.best_loss, self.epochs_since_best = val_loss, 0
        else:
            self.epochs_since_best += 1

    @property
    def should_stop(self):
        """True once the validation loss has not decreased for `patience` epochs."""
        return bool(self.patience) and self.epochs_since_best >= self.patience


def train(dataloaders, model, criterion, optimizer, scheduler, device, config):
    """
    Train and validate the model over a given number of epochs.

    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk. When the scheduler lowers the
    learning rate, the best weights so far are reloaded. If config.early_stopping_patience is set,
    training stops once the validation loss has not decreased for that many epochs in a row.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        model (torch.nn.Module): The video classification model to be trained.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, which adjusts the learning rate based on
            validation loss.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        config (TrainConfig): Output folder, number of epochs, mixed precision and early stopping options.

    Returns:
        torch.nn.Module: The trained model loaded with the best validation weights.
    """
    best = BestModelTracker(model, config.output_dir)
    stopper = EarlyStopping(config.early_stopping_patience)
    scaler = torch.amp.GradScaler(device.type, enabled=config.amp)

    for epoch in range(1, config.n_epochs + 1):
        lrs = get_learning_rates(optimizer)
        print(f'Epoch {epoch}/{config.n_epochs}; Current learning rate(s) {lrs}')

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model, criterion, dataloaders['train'], device, optimizer,
                                                    scaler, config.amp)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, criterion, dataloaders['val'], device, amp=config.amp)
        best.update(model, val_accuracy, epoch)
        stopper.update(val_loss)

        # Update learning rate based on validation loss; restart from the best weights when it drops
        scheduler.step(val_loss)
        if get_learning_rates(optimizer) != lrs:
            print('Loading best model weights!')
            best.restore(model)

        log_epoch(epoch, lrs, {
            'train/loss': train_loss,
            'train/accuracy': train_accuracy,
            'val/loss': val_loss,
            'val/accuracy': val_accuracy,
            'val/best_accuracy': best.best_accuracy,
            'val/best_loss': stopper.best_loss,
        })

        if stopper.should_stop:
            print(f'Early stopping at epoch {epoch}: validation loss has not decreased for '
                  f'{config.early_stopping_patience} epochs.')
            if wandb.run is not None:
                wandb.run.summary['early_stopped_epoch'] = epoch
            break

    # Load the best model weights before returning the model
    best.restore(model)
    return model


def log_epoch(epoch, lrs, metrics):
    """
    Log the metrics of an epoch to W&B (if a run is active) and print a summary line.

    Args:
        epoch (int): Epoch number (starting at 1).
        lrs (dict): Learning rate of each parameter group during the epoch.
        metrics (dict): Metrics keyed by their W&B name ('train/loss', 'val/accuracy', ...).
    """
    if wandb.run is not None:
        wandb.log({'epoch': epoch, **{'lr/' + name: lr for name, lr in lrs.items()}, **metrics})

    print(f"train loss: {metrics['train/loss']:.6f}, val loss: {metrics['val/loss']:.6f}, "
          f"accuracy: {100 * metrics['val/accuracy']:.2f}")
    print("-" * 60)
    print()

def get_learning_rates(optimizer):
    """
    Retrieve the current learning rate of every parameter group of the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rates.

    Returns:
        dict: Learning rate per parameter group, keyed by the group's 'name' (or its index).
    """
    return {str(group.get('name', i)): group['lr'] for i, group in enumerate(optimizer.param_groups)}

def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.

    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).

    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds

def get_batch_loss(criterion, output, target, optimizer=None, scaler=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).

    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. If None, no
            backpropagation is performed.
        scaler (torch.amp.GradScaler, optional): Gradient scaler for mixed precision training. If None,
            gradients are not scaled.

    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    return loss.item(), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None, scaler=None, amp=False):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update model weights during training.
        scaler (torch.amp.GradScaler, optional): Gradient scaler for mixed precision training.
        amp (bool, optional): Run the forward pass under automatic mixed precision. Default is False.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output.float(), y_batch, optimizer, scaler)

        running_loss += batch_loss
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy
