MODEL=$1
echo "Evaluate CLIP-Score for ${MODEL}"
IMG_PATH="main_exps_fid/${MODEL}"
CLIP_TEXT="main_exps_fid/clip_metrics/${MODEL}.txt"
DATA_LIST="./data/coco.json"
CUDA_VISIBLE_DEVICES=0 python3 src/eval_clip_score.py --img_dir $IMG_PATH --save_txt $CLIP_TEXT --data_list $DATA_LIST
