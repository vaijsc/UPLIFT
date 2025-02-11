
# # sbv2 8.1
XDG_CACHE_HOME="./checkpoints" CUDA_VISIBLE_DEVICES=0 python src/gen_sbv2.py \
                        --save-dir="main_exps/gen_sbv2_81" \
                        --pipeline-ckpt="thuanz123/swiftbrush"  \
                        --precision="fp16" \
                        --json="data/hpsv2" \
                        --seed=2710 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 32 \

# sbv1
# XDG_CACHE_HOME="./checkpoints" CUDA_VISIBLE_DEVICES=0 python src/gen_sbv1.py \
#                         --save-dir="main_exps/gen_sbv1" \
#                         --pipeline-ckpt="thuanz123/swiftbrush"  \
#                         --precision="fp16" \
#                         --json="data/hpsv2" \
#                         --seed=2710 \
#                         --start=0 \
#                         --end=-1 \
#                         --batch-size 32 \