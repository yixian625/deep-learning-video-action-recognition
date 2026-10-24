#!/bin/bash
# Run from the repo root (paths like data/ and models/ are relative to it)
cd "$(dirname "$0")/.."
# Frames: python src/preprocess.py -vd data/UCF50 -fd data/UCF50_clips -nf 32 --sampling dense --stride 2
python src/run.py \
    --frame_dir data/UCF50_clips \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type s3d \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --n_epochs 20 \
    --num_workers 6 \
    --amp \
    --mode train
