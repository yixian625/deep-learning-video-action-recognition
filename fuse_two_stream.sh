#!/bin/bash
# Late fusion of the RGB (test_s3d.sh) and flow (test_s3d_flow.sh) streams
python fuse.py \
    --preds models/s3d/test_predictions.npz models/s3d_flow/test_predictions.npz \
    --names rgb flow
