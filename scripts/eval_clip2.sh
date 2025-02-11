models=(
    # "coco_gen_instaflow_1step"
    # "coco_gen_lcm_1step"
    # "coco_gen_sd_turbo_1"
    # "gen_sbv1_viet"
    # "gen_sbv2_81_viet"
    # "enhance_by_dmd2"
    # "enhance_by_pixart_dmd"
    # "enhance_by_sdxl_lightning"
    # "enhance_by_turbo"
    # "sdxl_lightning"
    # "pixart_dmd"
    # "enhance_dmd2"
    # "enhance_pixart_dmd"
)

for MODEL in "${models[@]}"
do
    echo "Evaluate CLIP-Score for ${MODEL}"
    IMG_PATH="coco/gen/${MODEL}"
    CLIP_TEXT="coco/gen/clip_score/${MODEL}.txt"
    DATA_LIST="./data/coco.json"
    CUDA_VISIBLE_DEVICES=0 python3 src/eval_clip_score.py --img_dir $IMG_PATH --save_txt $CLIP_TEXT --data_list $DATA_LIST
done
