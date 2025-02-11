# CUDA_VISIBLE_DEVICES=0 python gen_scripts/image_enhance.py \
#                               sbv2_path=checkpoints/ckpt/sb_v2_ckpt/0.5 \
#                               sbv2_inverse_path=checkpoints/ckpt/inverse2enhance_ckpt/checkpoint-10000 \
#                               prompt_path=data/hpsv2/hpsv2_benchmark_anime.json \
#                               save_dir=experiments/enhance_gen_LCM_1step/hpsv2_benchmark_anime \
#                               image_folder=experiments/gen_LCM_1step/hpsv2_benchmark_anime \

CUDA_VISIBLE_DEVICES=0 python gen_scripts/image_enhance.py \
                              sbv2_path=checkpoints/ckpt/sb_v2_ckpt/0.5 \
                              sbv2_inverse_path=checkpoints/ckpt/inverse2enhance_ckpt/checkpoint-10000 \
                              prompt_path=data/hpsv2/hpsv2_benchmark_paintings.json \
                              save_dir=experiments/enhance_gen_LCM_1step/hpsv2_benchmark_paintings \
                              image_folder=experiments/gen_LCM_1step/hpsv2_benchmark_paintings \

CUDA_VISIBLE_DEVICES=0 python gen_scripts/image_enhance.py \
                              sbv2_path=checkpoints/ckpt/sb_v2_ckpt/0.5 \
                              sbv2_inverse_path=checkpoints/ckpt/inverse2enhance_ckpt/checkpoint-10000 \
                              prompt_path=data/hpsv2/hpsv2_benchmark_photo.json \
                              save_dir=experiments/enhance_gen_LCM_1step/hpsv2_benchmark_photo \
                              image_folder=experiments/gen_LCM_1step/hpsv2_benchmark_photo \

CUDA_VISIBLE_DEVICES=0 python gen_scripts/image_enhance.py \
                              sbv2_path=checkpoints/ckpt/sb_v2_ckpt/0.5 \
                              sbv2_inverse_path=checkpoints/ckpt/inverse2enhance_ckpt/checkpoint-10000 \
                              prompt_path=data/hpsv2/hpsv2_benchmark_concept_art.json \
                              save_dir=experiments/enhance_gen_LCM_1step/hpsv2_benchmark_concept_art \
                              image_folder=experiments/gen_LCM_1step/hpsv2_benchmark_concept_art \
