#!/bin/bash

accelerate launch -m train_scripts.train_s1_l2xz_dis_skip \
    --path_prompt_train="data/40k_mapping.json" \
    --pretrained_sb_generator="checkpoints/sbv2_fid81/unet" \
    --pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1-base" \
    --use_ema \
    --resolution 512 \
    --validation_steps 200 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
	--set_grads_to_none \
    --guidance_scale 4.5 \
    --learning_rate 1.e-05 \
    --lr_scheduler cosine \
	--adam_weight_decay 1.e-04 \
	--lr_warmup_steps 0 \
    --num_train_epochs 200 \
    --checkpointing_steps 1000 \
	--output_dir "training-runs/train_inverse_from_fid81_3l2z_1l2x_fp32_dis_skip" \
    --data_option "None" \
    --w_l2z=3.0 \
    --w_l2x=1.0 \
