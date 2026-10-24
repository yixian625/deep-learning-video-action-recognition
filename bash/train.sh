#!/bin/bash
# Run from the repo root (paths like data/ and models/ are relative to it)
cd "$(dirname "$0")/.."
# LRCN baseline: original hyperparameters (run.py defaults: ResNet-34, LSTM 1x100, dropout 0.1,
# Adam lr 3e-5, 30 epochs), with all bug fixes; --num_workers only speeds up data loading
python src/run.py \
    --frame_dir data/UCF50_frames \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --num_workers 6 \
    --run_name lrcn-baseline-train \
    --mode train
