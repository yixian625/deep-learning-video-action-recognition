#!/bin/bash
# Flow: python compute_flow.py --clip_dir data/UCF50_clips --flow_dir data/UCF50_flow --amp
# Head (final 1x1x1 conv) lr 5e-4, backbone lr 5e-5; stops when val loss has not decreased for 10 epochs
python run.py \
    --frame_dir data/UCF50_flow \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type s3d_flow \
    --n_classes 50 \
    --fr_per_vid 31 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --backbone_lr 5e-5 \
    --dropout 0.2 \
    --n_epochs 100 \
    --early_stopping_patience 10 \
    --num_workers 6 \
    --amp \
    --run_name s3d-two-flow-train \
    --mode train
