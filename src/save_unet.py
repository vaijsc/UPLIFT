from lib.swiftbrush import SBV2Gen, SBV2Inverse
import torch
import os
sb_gen = SBV2Gen("checkpoints/ckpt/sb_v2_ckpt/0.5")
sb_inverted = SBV2Inverse("training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000")

save_path = "_saved_model"
os.makedirs(save_path, exist_ok=True)
torch.save(sb_gen.unet, save_path+'/sbgen_unet.pth')
torch.save(sb_inverted.unet, save_path+'/inverted_unet.pth')

breakpoint()