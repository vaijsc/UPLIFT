CUDA_VISIBLE_DEVICES=0 python src/gen_sbv2_user_study.py \
                        --save-dir="main_exps/user_study_sbv2" \
                        --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
                        --precision="fp16" \

chmod 777 -R main_exps/user_study_sbv2