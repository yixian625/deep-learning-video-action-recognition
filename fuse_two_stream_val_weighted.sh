#!/bin/bash
# Late fusion with the stream weights chosen on the validation set (not the test set):
# 1) evaluate both streams on the validation split (writes val_predictions.npz next to each checkpoint),
# 2) grid-search the flow weight on validation, then apply it to the test predictions.
# Requires test_s3d_rgb.sh and test_s3d_flow.sh to have been run (test_predictions.npz).
set -e
python run.py --ckpt models/s3d_two_rgb/best_model_wts.pt --model_type s3d --n_classes 50 --fr_per_vid 32 \
    --batch_size 8 --num_workers 6 --amp --eval_split val --run_name s3d-two-rgb-val --mode eval
python run.py --ckpt models/s3d_flow/best_model_wts.pt --model_type s3d_flow --n_classes 50 --fr_per_vid 32 \
    --batch_size 8 --num_workers 6 --amp --eval_split val --run_name s3d-two-flow-val --mode eval
python fuse.py \
    --preds models/s3d_two_rgb/test_predictions.npz models/s3d_flow/test_predictions.npz \
    --val_preds models/s3d_two_rgb/val_predictions.npz models/s3d_flow/val_predictions.npz \
    --names rgb flow \
    --run_name s3d-two-fusion-val-weighted
