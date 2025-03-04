CUDA_VISIBLE_DEVICES=0 python src/gen_flux.py \
                        --save-dir="main_exps/gen_flux1_dev_50steps" \
                        --pipeline-ckpt="/lustre/scratch/client/vinai/users/hungnm66/Projects/pretrained_sd_models/FLUX.1-dev"  \
                        --precision="fp16" \
                        --json="data/hpsv2" \
                        --seed=2710 \
                        --inference-steps=50 \
                        --guidance-scale=3.5 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 2 \

# CUDA_VISIBLE_DEVICES=0 python src/gen_flux.py \
#                         --save-dir="main_exps/gen_flux1_schnell_4steps" \
#                         --pipeline-ckpt="black-forest-labs/FLUX.1-schnell"  \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=4 \
#                         --guidance-scale=0.0 \
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


# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/gen_sdv14" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \