#!/bin/bash

# 823
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_freeu.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/PAG_gen_sdv21_base_debug" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=200 \
#                         --batch-size 1 \
#                         --b1=1.5 \
#                         --b2=1.6 \
#                         --s1=0.9 \
#                         --s2=0.2 \
# 1128
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_pag.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/PAG_gen_sdv21_base_debug" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=2.0 \
#                         --start=0 \
#                         --end=200 \
#                         --batch-size 1 \
# 980
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_sag.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/SAG_gen_sdv21_base_debug" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=2.0 \
#                         --start=0 \
#                         --end=200 \
#                         --batch-size 1 \
#1546
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_uplift.py \
#                         --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
#                         --save-dir="main_exps/UPLIFT_gen_sdv21_base" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=2.0 \
#                         --start=0 \
#                         --end=200 \
#                         --batch-size 1 \
# 998