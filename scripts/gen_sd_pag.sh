# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_2.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/gen_sdv21_base_50steps" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=50 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \



# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_2.py \
#                         --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
#                         --save-dir="main_exps/gen_sdv15_50steps" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=50 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \


CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_pag.py \
                        --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
                        --save-dir="main_exps/PAG_gen_sdv15" \
                        --precision="fp16" \
                        --json="data/hpsv2" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=2.0 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 32 \


CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_pag.py \
                        --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
                        --save-dir="main_exps/PAG_gen_sdv21_base" \
                        --precision="fp16" \
                        --json="data/hpsv2" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=2.0 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 32 \