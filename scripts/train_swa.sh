#!/bin/bash
CUDA_VISIBLE_DEVICES=0 accelerate launch -m train_scripts.train_swa \
    --path_prompt_train="data/40k_mapping.json" \
    --pretrained_sb_generator="checkpoints/sbv2_fid81/unet" \
    --pretrained_model_name_or_path="checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c" \
    --use_ema \
    --resolution 512 \
    --validation_steps 200 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 --gradient_checkpointing \
	--set_grads_to_none \
    --guidance_scale 4.5 \
    --learning_rate 1.e-05 \
    --lr_scheduler cosine \
	--adam_weight_decay 1.e-04 \
	--lr_warmup_steps 0 \
    --num_train_epochs 200\
    --checkpointing_steps 1000 \
	--output_dir "training-runs/train_z2_x1_swa_f_100" \
    --data_option "None" \
    --w-l2z=2.0 \
    --w-l2x=1.0 \
    --max_train_steps 8000

    #	--output_dir "training-runs/train_inverse_from_fid81_10l2z_15l2x_sam" \
