#!/bin/bash
python run.py \
    --ckpt models/lrcn/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --num_workers 6 \
    --run_name lrcn-baseline-test \
    --mode eval
