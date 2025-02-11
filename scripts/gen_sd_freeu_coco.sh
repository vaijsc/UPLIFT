#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_freeu_coco.py \
#                         --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
#                         --save-dir="main_exps_fid/coco_FreeU_gen_sdv15" \
#                         --precision="fp16" \
#                         --json="data/coco_mapping_clean_15.json" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=100 \
#                         --batch-size 1 \
#                         --b1=1.5 \
#                         --b2=1.6 \
#                         --s1=0.9 \
#                         --s2=0.2 \
# #89.4
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_pag_coco.py \
#                        --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
#                        --save-dir="main_exps_fid/coco_PAG_gen_sdv15" \
#                        --precision="fp16" \
#                        --json="data/coco_mapping_clean_15.json" \
#                        --seed=2710 \
#                        --inference-steps=25 \
#                        --guidance-scale=2.0 \
#                        --start=0 \
#                        --end=100 \
#                        --batch-size 1 \

# #104.1
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_sag_coco.py \
#                        --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
#                        --save-dir="main_exps_fid/coco_PAG_gen_sdv15" \
#                        --precision="fp16" \
#                        --json="data/coco_mapping_clean_15.json" \
#                        --seed=2710 \
#                        --inference-steps=25 \
#                        --guidance-scale=2.0 \
#                        --start=0 \
#                        --end=100 \
#                        --batch-size 1 \
# #160.2
CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_uplift_coco.py \
                        --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
                        --save-dir="main_exps_fid/coco_uplift_gen_sdv15" \
                        --precision="fp16" \
                        --json="data/coco_mapping_clean_15.json" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \
                        --start=0 \
                        --end=100 \
                        --batch-size 1 \
                        --b1=1.5 \
                        --b2=1.6 \
                        --s1=0.9 \
                        --s2=0.2 \
#104.3                       
CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_freeu_coco.py \
                        --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
                        --save-dir="main_exps_fid/coco_FreeU_gen_sdv15" \
                        --precision="fp16" \
                        --json="data/coco_mapping_clean_15.json" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \
                        --start=0 \
                        --end=100 \
                        --batch-size 1 \
                        --b1=1.5 \
                        --b2=1.6 \
                        --s1=0.9 \
                        --s2=0.2 \
#131