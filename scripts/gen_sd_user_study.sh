CUDA_VISIBLE_DEVICES=0 python src/gen_user_study.py \
                        --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
                        --save-dir="main_exps/user_study_sd21_2" \
                        --precision="fp16" \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \

chmod 777 -R main_exps/user_study_sd21