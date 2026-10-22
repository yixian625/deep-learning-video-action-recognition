#!/bin/bash
# RGB stream of the two-stream model: same settings as train_s3d_flow.sh
# (train_s3d.sh is the original single-stream S3D run: lr 1e-4, 20 epochs)
python run.py \
    --frame_dir data/UCF50_clips \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type s3d \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --backbone_lr 5e-5 \
    --dropout 0.2 \
    --n_epochs 50 \
    --early_stopping_patience 10 \
    --num_workers 6 \
    --amp \
    --output_dir models/s3d_two_rgb \
    --run_name s3d-two-rgb-train \
    --mode train
