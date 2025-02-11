#   InstaFlow 1 step
CUDA_VISIBLE_DEVICES=0 python gen_scripts/lcm.py \
                        --json="data/hpsv2/" \
                        --save-dir="experiments/gen_lcm_1step" \
                        --precision="fp16" \
                        --seed=6969 \
                        --start=0 \
                        --inference-steps=1 \
                        --batch-size=16 \

#   InstaFlow 2 step
CUDA_VISIBLE_DEVICES=0 python gen_scripts/lcm.py \
                        --json="data/hpsv2/" \
                        --save-dir="main_exps/gen_lcm_2step" \
                        --precision="fp16" \
                        --seed=6969 \
                        --start=0 \
                        --inference-steps=2 \
                        --batch-size=16 \

#   Run instaflow 3 steps
CUDA_VISIBLE_DEVICES=0 python gen_scripts/lcm.py \
                        --json="data/hpsv2/" \
                        --save-dir="main_exps/gen_lcm_3step" \
                        --precision="fp16" \
                        --seed=6969 \
                        --start=0 \
                        --inference-steps=3 \
                        --batch-size=16 \


#   Run instaflow 4 steps
CUDA_VISIBLE_DEVICES=0 python gen_scripts/lcm.py \
                        --json="data/hpsv2/" \
                        --save-dir="experiments/gen_lcm_4step" \
                        --precision="fp16" \
                        --seed=6969 \
                        --start=0 \
                        --inference-steps=4 \
                        --batch-size=16 \

#   SD-Turbo 1 step
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_turbo.py \
#                         --pipeline-ckpt="checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"  \
#                         --save-dir="main_exps/gen_sd_turbo_1step" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=1 \
#                         --guidance-scale=0 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 16 \

# #   SD-Turbo 2 step
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_turbo.py \
#                         --pipeline-ckpt="checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"  \
#                         --save-dir="main_exps/gen_sd_turbo_2step" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=2 \
#                         --guidance-scale=0 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 16 \

# #   SD-Turbo 3 step
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_turbo.py \
#                         --pipeline-ckpt="checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"  \
#                         --save-dir="main_exps/gen_sd_turbo_3step" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=3 \
#                         --guidance-scale=0 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 16 \

# #   SD-Turbo 4 step
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_turbo.py \
#                         --pipeline-ckpt="checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"  \
#                         --save-dir="main_exps/gen_sd_turbo_4step" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=4 \
#                         --guidance-scale=0 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 16 \

# #   Addtional missing experiment from multistep
# CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12.py \
#                         --pipeline-ckpt="checkpoints/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"  \
#                         --save-dir="main_exps/gen_sdv14" \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --inference-steps=25 \
#                         --guidance-scale=7.5 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \