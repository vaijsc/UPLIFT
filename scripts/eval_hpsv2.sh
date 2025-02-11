#!/bin/bash

CUDA_VISIBLE_DEVICES=0  python src/eval_hpsv2.py \
                        --json="data/hpsv2" \
                        --score="hpsv2" \
                        --image-dir="main_exps/enhance_sd21_gen_sbv2_81" \
                        --save-res-dir="main_exps/enhance_sd21_gen_sbv2_81/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_instaflow_2step" \
#                         --save-res-dir="main_exps/gen_instaflow_2step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_instaflow_3step" \
#                         --save-res-dir="main_exps/gen_instaflow_3step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_instaflow_4step" \
#                         --save-res-dir="main_exps/gen_instaflow_4step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sd_turbo_1step" \
#                         --save-res-dir="main_exps/gen_sd_turbo_1step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sd_turbo_2step" \
#                         --save-res-dir="main_exps/gen_sd_turbo_2step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sd_turbo_3step" \
#                         --save-res-dir="main_exps/gen_sd_turbo_3step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sd_turbo_4step" \
#                         --save-res-dir="main_exps/gen_sd_turbo_4step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sdv14" \
#                         --save-res-dir="main_exps/gen_sdv14/hpsv2_results" \


# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sdv15" \
#                         --save-res-dir="main_exps/gen_sdv15/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sdv21_base" \
#                         --save-res-dir="main_exps/gen_sdv21_base/hpsv2_results" \


# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sbv1" \
#                         --save-res-dir="main_exps/gen_sbv1/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_sbv2_81" \
#                         --save-res-dir="main_exps/gen_sbv2_81/hpsv2_results" \


# CUDA_VISIBLE_DEVICES=0  python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_lcm_1step" \
#                         --save-res-dir="main_exps/gen_lcm_1step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_lcm_2step" \
#                         --save-res-dir="main_exps/gen_lcm_2step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_lcm_3step" \
#                         --save-res-dir="main_exps/gen_lcm_3step/hpsv2_results" \

# CUDA_VISIBLE_DEVICES=0 python src/eval_hpsv2.py \
#                         --json="data/hpsv2" \
#                         --score="hpsv2" \
#                         --image-dir="main_exps/gen_lcm_4step" \
#                         --save-res-dir="main_exps/gen_lcm_4step/hpsv2_results" \

# models=(
#     # "enhance_gen_lcm_1step"
#     # "enhance_gen_lcm_2step"
#     # "enhance_gen_lcm_3step"
#     # "enhance_gen_lcm_4step"
#     # "enhance_gen_sdv15"
#     # "enhance_gen_sdv21_base"
#     # "enhance_gen_instaflow_1step"
#     # "gen_sdv21_base_50steps"
#     # "gen_sdv15_50steps"
#     # "gen_sdv14_50steps"
#     # "enhance_gen_sd_turbo_1step"
#     # "enhance_gen_sd_turbo_2step"
#     # "enhance_gen_sd_turbo_3step"
#     # "enhance_gen_sd_turbo_4step"
#     # "gen_sdxl"
#     # "enhance_gen_sbv1"
#     # "enhance_gen_sbv2_81"
#     # "gen_sbv1"
#     # "enhance_gen_sdxl"
#     "SAG_gen_sdv15"
#     "PAG_gen_sdv15"
#     "PAG_gen_sdv21_base"
#     "SAG_gen_sdv21_base"
# )

# for MODEL in "${models[@]}"
# do
#     echo "Evaluate HPSV2 for ${MODEL}"
#     CUDA_VISIBLE_DEVICES=0 python -m src.eval_hpsv2 \
#                             --json="data/hpsv2" \
#                             --score="hpsv2" \
#                             --image-dir="main_exps/$MODEL" \
#                             --save-res-dir="main_exps/$MODEL/hpsv2_results" \

# done

# models=(
#     # fid81_10l2z_15l2x_2000
#     # fid81_10l2z_15l2x_4000
#     # fid81_10l2z_15l2x_6000
#     # fid81_10l2z_15l2x_8000
#     # fid81_2l2z_1l2x_fp32_6000
#     # fid81_2l2z_1l2x_fp32_7000
#     # fid81_2l2z_1l2x_fp32_8000
#     # fid81_2l2z_1l2x_fp32_9000
#     old_oldsbv2gen

# )
# for MODEL in "${models[@]}"
# do
#     echo "Evaluate HPSV2 for ${MODEL}"
#     XDG_CACHE_HOME=/lustre/scratch/client/vinai/users/ngannh9/.cache/ CUDA_VISIBLE_DEVICES=0 python -m src.eval_hpsv2_ngan \
#                             --json="data/hpsv2" \
#                             --image_dir="pickckpt/genhps/$MODEL" \
#                             --seed=2710 \

# done