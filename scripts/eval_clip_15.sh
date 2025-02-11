models=(
    # "coco_gen_instaflow_1step"
    # "coco_gen_lcm_1step"
    # "coco_gen_sd_turbo_1"
    # "gen_sbv1_viet"
    # "gen_sbv2_81_viet"
    # "enhance_coco_gen_instaflow_1step"
    # "enhance_coco_gen_sd_turbo_1"
    # "enhance_coco_gen_lcm_1step"
    # "enhance_gen_sbv1_viet"
    # "enhance_gen_sbv2_81_viet"
    # "sd15_25step"
    "enhance_sd15_25step"
)

for MODEL in "${models[@]}"
do
    echo "Evaluate CLIP-Score for ${MODEL}"
    IMG_PATH="main_exps_fid/${MODEL}"
    CLIP_TEXT="main_exps_fid/clip_metrics/${MODEL}.txt"
    DATA_LIST="./data/coco_30k_15.csv"
    XDG_CACHE_HOME=/lustre/scratch/client/vinai/users/ngannh9/.cache/ CUDA_VISIBLE_DEVICES=0 python3 src/eval_clip_15.py --img_dir $IMG_PATH --save_txt $CLIP_TEXT --data_list $DATA_LIST
done
