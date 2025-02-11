CUDA_VISIBLE_DEVICES=0 python test_concat_to_enhance.py \
                        sbv2_path=checkpoints/ckpt/sb_v2_ckpt/0.5/ \
                        sbv2_inverse_path=checkpoints/ckpt/inverse2enhance_ckpt/checkpoint-10000/ \
                        prompt_path=data/hps/prompt_anime.txt \
                        save_dir=$1 \
                        bsz=8