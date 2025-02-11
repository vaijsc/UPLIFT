stats=".cache/recalltf/coco_val256crop_savepng"
image_dir=$1
save_dir=$2

CUDA_VISIBLE_DEVICES=0 python eval/eval_precision.py $stats $image_dir
