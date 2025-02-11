SBV2=/data/quangnh24/checkpoints/datasets--nguyenquivinhquang--quang_dau_cac/snapshots/56400f2e7c14405248b7b56378d46cbe9d62eddb/sbv2_fid81/unet
SD21="/data/quangnh24/checkpoints/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_scripts/train_s1_l2xz.py \
    --path_prompt_train="40k_mapping.json" \
    --pretrained_sb_generator=${SBV2} \
    --pretrained_model_name_or_path=${SD21} \
    --resume_from_checkpoint "latest" \
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
    --num_train_epochs 100 \
    --checkpointing_steps 1000 \
	--output_dir "/data/quangnh24/exps_rebuttal/training-runs/train_ablate_1l2x_0.3l2z" \
    --data_option "None" \
    --w-l2z=3.0 \
    --w-l2x=1.0 \
