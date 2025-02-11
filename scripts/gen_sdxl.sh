# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/gen_sdv21_base" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \



# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12.py \
#                         --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
#                         --save-dir="main_exps/gen_sdv15" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \

export XDG_CACHE_HOME="./checkpoints/huggingface/hub"
CUDA_VISIBLE_DEVICES=0 python src/gen_sdxl.py \
                        --pipeline-ckpt="checkpoints/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"  \
                        --save-dir="main_exps/gen_sdxl" \
                        --precision="fp16" \
                        --json="data/hpsv2" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 8 \