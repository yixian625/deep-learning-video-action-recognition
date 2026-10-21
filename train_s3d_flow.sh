#!/bin/bash
# Flow: python compute_flow.py --clip_dir data/UCF50_clips --flow_dir data/UCF50_flow --amp
python run.py \
    --frame_dir data/UCF50_flow \
    --train_size 0.8 \
    --test_size 0.1 \
    --model_type s3d_flow \
    --n_classes 50 \
    --fr_per_vid 31 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --n_epochs 20 \
    --num_workers 6 \
    --amp \
    --mode train
