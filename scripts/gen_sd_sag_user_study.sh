CUDA_VISIBLE_DEVICES=0 python src/gen_user_study_sag.py \
                        --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
                        --save-dir="main_exps/user_study_sdv21_sag" \
                        --precision="fp16" \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \
                        # --start=0 \
                        # --end=-1 \
                        # --batch-size 32 \