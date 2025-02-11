#!/bin/bash
exps_name=(
    # "writting/exps/ablation_hps/enhance_gen_instaflow_1step_by_/se_from_fid81_0l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_old/se_from_fid81_2l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_old"
    # "writting/exps/ablation_hps/enhance_gen_instaflow_1step_by_old/se_from_fid81_2l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_instaflow_1step_by_/se_from_fid81_1l2z_0l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_turbo/fid81_1l2z_1l2x_fp32_gener_turbo"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_/se_from_fid81_0l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_/se_from_fid81_1l2z_0l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_old/se_from_fid81_2l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_/se_from_fid81_0l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_/se_from_fid81_1l2z_0l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_/se_from_fid81_0l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_/se_from_fid81_0l2z_1l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_instaflow_1step_by_turbo/fid81_1l2z_1l2x_fp32_gener_turbo"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_turbo/fid81_1l2z_1l2x_fp32_gener_turbo"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_/se_from_fid81_1l2z_0l2x_fp32"
    # "writting/exps/ablation_hps/enhance_gen_instaflow_1step_by_turbo/fid81_1l2z_1l2x_fp32_gener_turbo"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_turbo/fid81_1l2z_1l2x_fp32_gener_turbo"
    # "writting/exps/ablation_hps/enhance_gen_sd_turbo_1step_by_/_swa"
    # "writting/exps/ablation_hps/enhance_gen_sbv2_81_by_/_swa"
    # "writting/exps/ablation_hps/enhance_gen_sdv21_base_by_/_swa"
    # "_genhps_ddim/enhance_gen_instaflow_1step_10_step"
    # "_genhps_ddim/enhance_gen_sbv2_81_10_step"
    # "_genhps_ddim/enhance_gen_sdv21_base_10_step"
    # "_genhps_ddim/enhance_gen_instaflow_1step_30_step"
    # "_genhps_sd21_10step"
    "_enhhps_sd21_10step"

)
for EXPS_NAME in "${exps_name[@]}"
do
    echo "Evaluate HPSV2 for ${EXPS_NAME}"
    XDG_CACHE_HOME=/lustre/scratch/client/vinai/users/ngannh9/.cache/ CUDA_VISIBLE_DEVICES=0 python -m src.eval_hpsv2_ngan \
        --json="data/hpsv2" \
        --seed=2710 \
        --image_dir="${EXPS_NAME}" \
        --out_dir="${EXPS_NAME}/_results"

done
