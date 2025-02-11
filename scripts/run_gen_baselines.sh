#   InstaFlow 1 step
CUDA_VISIBLE_DEVICES=0 python gen_scripts/instaflow.py \
                        --json="data/hpsv2/" \
                        --save-dir="experiments/lcm_1step_gen" \
                        --precision="fp16" \
                        --seed=2710 \
                        --start=0 \
                        --inference-steps=1 \
                        --batch-size=20 \

#   Run instaflow 3 steps
CUDA_VISIBLE_DEVICES=0 python gen_scripts/instaflow.py \
                        --json="data/hpsv2/" \
                        --save-dir="experiments/lcm_3step_gen" \
                        --precision="fp16" \
                        --seed=2710 \
                        --start=0 \
                        --inference-steps=3 \
                        --batch-size=20 \

#   Run instaflow 4 steps
CUDA_VISIBLE_DEVICES=0 python gen_scripts/instaflow.py \
                        --json="data/hpsv2/" \
                        --save-dir="experiments/lcm_4step_gen" \
                        --precision="fp16" \
                        --seed=2710 \
                        --start=0 \
                        --inference-steps=4 \
                        --batch-size=20 \



