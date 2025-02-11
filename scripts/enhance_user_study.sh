CUDA_VISIBLE_DEVICES=0 python gen_scripts/enhance_user.py \
                              sbv2_path=checkpoints/sbv2_fid81/unet \
                              sbv2_inverse_path=training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000 \
                              image_folder=main_exps/user_study_sd21_2 \

