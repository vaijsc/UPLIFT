UNET_SB="checkpoints/sbv2_fid81/unet" 
PIXART="checkpoints/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692"

accelerate launch train_scripts/train_s1_l2xz_pixart_2.py \
    --path_prompt_train="" \
    --pretrained_sb_generator=${PIXART} \
    --pretrained_model_name_or_path=${PIXART} \
    --resume_from_checkpoint "" \
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
    --checkpointing_steps 2000 \
	--output_dir "training-runs/train_inverse_from_fid81_1l2z_1l2x_fp32_pixart_same_arch" \
	--embed_folder "data/journeydb_features" \
	--json_path "data/journeydb_50k.json" \
    --data_option "None" \
    --w-l2z=1.0 \
    --w-l2x=1.0 \
    # --use_ema \