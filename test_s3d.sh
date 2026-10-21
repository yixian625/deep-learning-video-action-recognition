#!/bin/bash
python run.py \
    --ckpt models/s3d/best_model_wts.pt \
    --model_type s3d \
    --n_classes 50 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --num_workers 6 \
    --amp \
    --mode eval
