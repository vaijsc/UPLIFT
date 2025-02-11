CUDA_VISIBLE_DEVICES=0 python test_degradation.py \
                              sbv2_path=checkpoints/sbv2_fid81/unet \
                              sbv2_inverse_path=training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000 \
                              prompt_path=data/hps_enhance/prompt_anime.txt \
                              bsz=16 \
                              save_dir=experiments/test_fp32_ckpt2k
