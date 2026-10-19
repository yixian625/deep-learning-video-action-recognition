#!/bin/bash
python run.py \
    --frame_dir data/UCF50_frames \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode 'train'