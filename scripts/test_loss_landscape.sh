# res=$1
# guidance=(
#     0.1
#     0.2
#     0.3
#     0.4
#     0.5
#     0.6
#     0.7
#     0.8
#     0.9
#     1.0
# )

# for gs in "${guidance[@]}"
# do
#     echo "Guidance: ${gs} -- Resolution: ${res}"
#     CUDA_VISIBLE_DEVICES=0 python test_degradation.py \
#                                 sbv2_path=checkpoints/sbv2_fid81/unet \
#                                 sbv2_inverse_path=training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000 \
#                                 prompt_path=data/hps_enhance/prompt_paintings.txt \
#                                 bsz=8 \
#                                 save_dir=test_controllable/test_mask_image_gs${gs}_res${res}_paintings_down \
#                                 normalize_noise=False \
#                                 guidance=$gs \
#                                 resolution=$res
# done 
CUDA_VISIBLE_DEVICES=0 python test_attention.py \
                            sbv2_path=checkpoints/sbv2_fid81/unet \
                            sbv2_inverse_path=training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000 \
                            prompt_path=data/hps_enhance/prompt_anime.txt \
                            bsz=1 \
                            save_dir=test_controllable/test_inject_background_attention_base \
                            control_attention=False \