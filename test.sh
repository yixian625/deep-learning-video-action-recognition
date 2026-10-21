#!/bin/bash
python run.py \
    --ckpt models/lrcn/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 50 \
    --model_type lrcn \
    --batch_size 4 \
    --mode eval
