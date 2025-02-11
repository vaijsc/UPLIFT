#!/bin/bash
# models=(
#     # "coco_gen_instaflow_1step"
#     # "coco_gen_lcm_1step"
#     # "coco_gen_sd_turbo_1"
#     # "gen_sbv1_viet"
#     # "gen_sbv2_81_viet"
#     # "enhance_coco_gen_instaflow_1step"
#     # "enhance_coco_gen_sd_turbo_1"
#     # "enhance_coco_gen_lcm_1step"
#     # "enhance_gen_sbv1_viet"
#     # "enhance_gen_sbv2_81_viet"
#     # "sd15_25step"
#     # "enhance_sd15_25step"
# )

# for MODEL in "${models[@]}"
# do
#     echo "Evaluate CMMD for ${MODEL}"
#     IMG_PATH="main_exps_fid/${MODEL}"
#     CUDA_VISIBLE_DEVICES=0 python cmmd_pytorch/main.py data/val_fid2014/mscoco_val2014 main_exps_fid/${MODEL} --batch_size=256 --max_count=30000 > main_exps_fid/cmmd_metrics/${MODEL}.txt
# done

folder=coco_PAG_gen_sdv21
echo "CMMD for ${folder}"
CUDA_VISIBLE_DEVICES=0 python cmmd_pytorch/main.py data/val_fid2014/mscoco_val2014 main_exps_fid/${folder} --batch_size=256 --max_count=30000 > main_exps_fid/cmmd_metrics/${folder}.txt
