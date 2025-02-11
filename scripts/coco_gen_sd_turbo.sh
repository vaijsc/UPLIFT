#   InstaFlow 1 step
CUDA_VISIBLE_DEVICES=0 python src/coco_gen_sd_turbo.py \
                            --pipeline-ckpt="checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"  \
                            --save-dir="main_exps_fid/coco_gen_sd_turbo_1" \
                            --precision="fp16" \
                            --json="data/coco_mapping_clean.json" \
                            --seed=2710 \
                            --inference-steps=1 \
                            --guidance-scale=0 \
                            --start=0 \
                            --end=-1 \
                            --batch-size 1 \

CUDA_VISIBLE_DEVICES=0 python src/enhance_coco.py \
                                sbv2_path="checkpoints/sbv2_fid81/unet" \
                                sbv2_inverse_path="training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000" \
                                save_dir="main_exps_fid/enhance_coco_gen_sd_turbo_1" \
                                image_folder="main_exps_fid/coco_gen_sd_turbo_1" \

