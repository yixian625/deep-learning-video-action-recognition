#!/bin/bash
# Run from the repo root (paths like data/ and models/ are relative to it)
cd "$(dirname "$0")/.."
python src/run.py \
    --ckpt models/s3d_two_rgb/best_model_wts.pt \
    --model_type s3d \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --num_workers 6 \
    --amp \
    --run_name s3d-two-rgb-test \
    --mode eval
