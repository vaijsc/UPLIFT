CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_pag_coco.py \
                        --pipeline-ckpt="checkpoints/stable-diffusion-2-1-base"  \
                        --save-dir="main_exps_fid/coco_PAG_gen_sd21" \
                        --precision="fp16" \
                        --json="data/coco_mapping_clean.json" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=2.0 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 32 \

