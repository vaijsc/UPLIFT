#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m coco.src.enhance \
        eval_models="training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-15000" \
        original_image_path=coco/gen/turbo_1 \
        seed=2710 \
        batch_size=32 \
        num=100
        # eval_models="training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-10000,training-runs/train_inverse_from_fid81_10l2z_15l2x/checkpoint-10000,checkpoints/ckpt/sb_v2_ckpt/0.5" \
