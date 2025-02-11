#   InstaFlow 1 step
CUDA_VISIBLE_DEVICES=0 python src/coco_gen_lcm.py \
                        --json="data/coco_mapping_clean.json" \
                        --save-dir="main_exps_fid/coco_gen_lcm_1step" \
                        --precision="fp16" \
                        --seed=6969 \
                        --start=0 \
                        --inference-steps=1 \
                        --batch-size=1 \


CUDA_VISIBLE_DEVICES=0 python src/enhance_coco.py \
                                sbv2_path="checkpoints/sbv2_fid81/unet" \
                                sbv2_inverse_path="training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000" \
                                save_dir="main_exps_fid/enhance_coco_gen_lcm_1step" \
                                image_folder="main_exps_fid/coco_gen_lcm_1step" \