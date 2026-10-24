#!/bin/bash
# Run from the repo root (paths like data/ and models/ are relative to it)
cd "$(dirname "$0")/.."
# Flow: python src/compute_flow.py --clip_dir data/UCF50_clips --flow_dir data/UCF50_flow --amp
# Head (final 1x1x1 conv) lr 5e-4, backbone lr 1e-4 (2x the RGB stream: flow differs more from the RGB pretraining)
# BatchNorm momentum 0.01 (torchvision default 0.001) so the running statistics adapt to flow
# 32 time steps (31 flow fields + the last one repeated): an odd T=31 is ~2.8x slower in cuDNN
# Stops when val loss has not decreased for 10 epochs
python src/run.py \
    --frame_dir data/UCF50_flow \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type s3d_flow \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --backbone_lr 1e-4 \
    --bn_momentum 0.01 \
    --dropout 0.2 \
    --n_epochs 50 \
    --early_stopping_patience 10 \
    --num_workers 6 \
    --amp \
    --run_name s3d-two-flow-train \
    --mode train
