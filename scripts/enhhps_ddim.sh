#!/bin/bash

gen_models=(
    # "gen_sd_turbo_1step"
    # "gen_sbv1"
    # "gen_lcm_1step"
    # "gen_sdv15"
    # "gen_sbv2_81"
    "gen_sdv21_base"
    # "gen_instaflow_1step"
)
# MODEL="training-runs/train_from_fid81_1l2z_1l2x_fp32_gener_turbo/checkpoint-6000"
# MODEL="training-runs/train_inverse_from_fid81_10l2z_15l2x/checkpoint-4000"
# MODEL="training-runs/train_inverse_from_fid81_1l2z_0l2x_fp32/checkpoint-8000"
# MODEL="training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000"
# MODEL="training-runs/train_z2_x1_swa/checkpoint-8000"


ENHANCE_GENERATOR=""
INVERT_STEP=30
# STYLE="photo"
STYLE="concept_art,anime,paintings,photo"
for GEN_MODEL in "${gen_models[@]}"
do
    save_path="_genhps_ddim/enhance_${GEN_MODEL}_${INVERT_STEP}_step"
    echo "saving to ${save_path}"
    XDG_CACHE_HOME=/lustre/scratch/client/vinai/users/ngannh9/.cache/ CUDA_VISIBLE_DEVICES=0 python -m pick_quan.enhhps_ddim \
                            save_path=${save_path} \
                            batch_size=1 \
                            original_image_path=main_exps/$GEN_MODEL \
                            enh_generator=${ENHANCE_GENERATOR} \
                            style=$STYLE \
                            start=0 \
                            end=800 \
                            invert_step=$INVERT_STEP
done
#original_image_path=main_exps/gen_sd_turbo_1step \
#
