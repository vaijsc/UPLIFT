Ngan_071020030CUDA_VISIBLE_DEVICES=0 python src/gen_user_study_freeu.py \
                        --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
                        --save-dir="main_exps/user_study_sd21_freeu" \
                        --precision="fp16" \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \
                        --b1=1.4 \
                        --b2=1.6 \
                        --s1=0.9 \
                        --s2=0.2 \
