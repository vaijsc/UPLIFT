import torch
import os.path as osp
import os
import argparse
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image, make_grid
from tqdm import tqdm 


@torch.no_grad()
def inference(vae, tokenizer, text_encoder,
              unet_inverse, noise_scheduler, sb_generator,
              src_prompt,
              weight_dtype, device):
    # prepare stuff
    T = torch.ones((1,), dtype=torch.int64, device=device)
    T = T * (noise_scheduler.config.num_train_timesteps - 1)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    alpha_t = (alphas_cumprod[T] ** 0.5).view(-1, 1, 1, 1)
    sigma_t = ((1 - alphas_cumprod[T]) ** 0.5).view(-1, 1, 1, 1)
    del alphas_cumprod
    
    input_shape = (1, 4, args.resolution // 8, args.resolution // 8)
    # gt_z = torch.randn(*input_shape, dtype=weight_dtype, device=accelerator.device)
    gt_z = gen_random_tensor_fix_seed(input_shape, args.seed, weight_dtype, device)
                        
    # Get groundtruth x
    final_timestep = torch.ones((1,), dtype=torch.int64, device=device)
    final_timestep = final_timestep * 999
    
    input_id = tokenize_captions(src_prompt, tokenizer).to(device)
    encoder_hidden_state = text_encoder(input_id)[0].to(device, dtype=weight_dtype)
    
    # get image gen by sb
    gt_x = sb_generator(gt_z, final_timestep, encoder_hidden_state).sample.to(dtype=weight_dtype)
    gt_x = (gt_z - sigma_t * gt_x) / alpha_t
    
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
    mid_timestep = mid_timestep * 500
    
    # predict inverted noise
    pred_z = unet_inverse(gt_x, mid_timestep, encoder_hidden_state).sample.to(dtype=weight_dtype)
    
    # predict reconstruct image
    pred_x = sb_generator(pred_z, final_timestep, encoder_hidden_state).sample.to(dtype=weight_dtype)
    
    # decode reconstruct image
    pred_original_latent = (pred_z - sigma_t * pred_x) / alpha_t
    pred_original_latent = pred_original_latent / vae.config.scaling_factor
    pred_image = (
        vae.decode(pred_original_latent.to(dtype=weight_dtype)).sample.float() + 1
    ) / 2
    
    # decode generated image
    gt_image = gt_x / vae.config.scaling_factor
    gt_image = (
        vae.decode(gt_image.to(dtype=weight_dtype)).sample.float() + 1
    ) / 2
    
    # decode inverted noise for visualization
    noise_img = pred_z / vae.config.scaling_factor
    noise_img = (
        vae.decode(noise_img.to(dtype=weight_dtype)).sample.float() + 1
    ) / 2
    
    return pred_image.detach().cpu()[0],\
            gt_image.detach().cpu()[0],\
                noise_img.detach().cpu()[0]

def main():
    