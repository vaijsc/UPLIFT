CUDA_VISIBLE_DEVICES=0 python src/gen_sd_v12_freeu.py \
                        --pipeline-ckpt="checkpoints/stable_diffusion_v15"  \
                        --save-dir="main_exps/FreeU_gen_sdv15" \
                        --precision="fp16" \
                        --json="data/hpsv2" \
                        --seed=2710 \
                        --inference-steps=25 \
                        --guidance-scale=7.5 \
                        --start=0 \
                        --end=-1 \
                        --batch-size 32 \
                        --b1=1.5 \
                        --b2=1.6 \
                        --s1=0.9 \
                        --s2=0.2 \