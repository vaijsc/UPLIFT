import torch
import os.path as osp
import json
import os
import numpy as np
import argparse
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image, make_grid
from tqdm import tqdm 
import random

class SBV2Gen():
    def __init__(self, path_ckpt_sbv2, model_name = "checkpoints/stable-diffusion-2-1-base"):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        self.unet_gen = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to("cuda")
        self.unet_gen.eval()

        self.last_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)
        self.first_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        
        # prepare stuff
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (self.alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
        # self.sigma_t = ((1 - self.alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)

    def get_alpha_t(self, timesteps):
        return (self.alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
    def get_sigma_t(self, timesteps):
        return ((1. - self.alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)

class SBV2Inverse():
    def __init__(self, path_ckpt, dtype="fp32"):
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32
        
        self.model_name = "checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae").to(
            "cuda", dtype=self.weight_dtype
        )

        self.unet_inverse = UNet2DConditionModel.from_pretrained(f"{path_ckpt}", subfolder="unet_ema").to(
            "cuda", dtype=self.weight_dtype
        )

        self.unet_inverse.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder"
        ).to("cuda", dtype=self.weight_dtype)
        
        self.mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.mid_timestep = self.mid_timestep * 500
        
        # prepare stuff
        T = torch.ones((1,), dtype=torch.int64, device="cuda")
        T = T * (self.noise_scheduler.config.num_train_timesteps - 1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[T] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[T]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
        
def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

def generate_image(sbv2_gen, noise, encoder_hidden_state, timesteps):
    model_pred = sbv2_gen.unet_gen(noise, timesteps, encoder_hidden_state).sample
    alpha_t = sbv2_gen.get_alpha_t(timesteps)
    sigma_t = sbv2_gen.get_sigma_t(timesteps)

    pred_original_sample = (noise - sigma_t * model_pred) / alpha_t
    if sbv2_gen.noise_scheduler.config.thresholding:
            pred_original_sample = sbv2_gen.noise_scheduler._threshold_sample(
            pred_original_sample
        )
    elif sbv2_gen.noise_scheduler.config.clip_sample:
        clip_sample_range = sbv2_gen.noise_scheduler.config.clip_sample_range
        pred_original_sample = pred_original_sample.clamp(
            -clip_sample_range, clip_sample_range
        )
    pred_original_sample = pred_original_sample / sbv2_gen.vae.config.scaling_factor
    refine_image = (sbv2_gen.vae.decode(pred_original_sample).sample + 1) / 2
    return refine_image

def compute_inverted_code(gen_image, sbv2_inverse, encoder_hidden_state):
    input_image = gen_image * 2 - 1
    latents = sbv2_inverse.vae.encode(input_image.to(torch.float32)).latent_dist.sample()
    latents = latents * sbv2_inverse.vae.config.scaling_factor
    predict_inverted_code = sbv2_inverse.unet_inverse(latents, sbv2_inverse.mid_timestep, encoder_hidden_state).sample.to(dtype=torch.float32)
    return predict_inverted_code

class SBV2Enhance():
    def __init__(self, path_sb_v2_gen_model, path_sb_v2_inverse_model):
        # define sb gen and inverse model
        self.sbv2_gen = SBV2Gen(path_sb_v2_gen_model)
        self.sbv2_inverse = SBV2Inverse(path_sb_v2_inverse_model)

    @torch.no_grad()
    def __call__(self, prompts, generator, noise):
        prompts, bsz = prompts
        total_image = len(prompts) if len(prompts) > 1 else bsz

        if len(prompts) == 1 and isinstance(prompts, list):
            prompts = prompts * bsz
        elif len(prompts) == 1:
            prompts = [prompts] * bsz

        if noise is None:
            noise = torch.randn(total_image, 4, 64, 64, device="cuda", generator=generator)

        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0]
        gen_image = generate_image(self.sbv2_gen, noise, encoder_hidden_state, timesteps=self.sbv2_gen.last_timestep)
        
        # Stage 2.1: Find inverted code
        predict_inverted_code = compute_inverted_code(gen_image, self.sbv2_inverse, encoder_hidden_state)

        # Stage 2.2: Gen img with inverted code
        return predict_inverted_code

def get_prompts(path):
    _, ext = path.split(".")
    
    if ext == "txt":
        f = open(path, "r")
        objs = f.read()
        prompts = objs.split("\n")
        return prompts
    
    elif ext == "json":
        return json.load(open(path, "r"))

def main(args):
    path_sbv2_gen = "checkpoints/sbv2_fid81/unet"

    inverse_paths = {
        "loss_combine": "training-runs/train_inverse_from_fid81_2l2z_1l2x_fp32/checkpoint-8000",
        "loss_x": "training-runs/train_inverse_from_fid81_0l2z_1l2x_fp32/checkpoint-8000",
        "loss_z": "training-runs/train_inverse_from_fid81_1l2z_0l2x_fp32/checkpoint-8000"
    }


    save_dir = "experiments/latent_codes"

    prompts = get_prompts("data/hpsv2/hpsv2_benchmark_anime.json")
    
    generator = torch.Generator("cuda").manual_seed(6969)

    init_gaussian = torch.randn(100,4,64,64)
    np.save(
        f"{save_dir}/init_gaussian.npy", init_gaussian
    )
    init_gaussian = init_gaussian.to("cuda")
    
    for prompt in tqdm(prompts[:64]):
        os.makedirs(f"{save_dir}/{prompt}/loss_combine", exist_ok=True)
        os.makedirs(f"{save_dir}/{prompt}/loss_x", exist_ok=True)
        os.makedirs(f"{save_dir}/{prompt}/loss_z", exist_ok=True)
        for key, path_inverse_net in inverse_paths.items():
            pred_inv_noise = []
            print(f"Init inversion from {path_inverse_net}")
            enhance_model = SBV2Enhance(path_sb_v2_gen_model=path_sbv2_gen, path_sb_v2_inverse_model=path_inverse_net)
            for noise in init_gaussian:
                input_prompts = [prompt]
                inverted_code = enhance_model((input_prompts, len(input_prompts)), generator=generator, noise=noise.unsqueeze(0))
                pred_inv_noise.append(inverted_code)

            pred_inv_np = torch.cat(pred_inv_noise).detach().cpu().permute(0,2,3,1).numpy()
            np.save(
                f"{save_dir}/{prompt}/{key}/inverted_code.npy", pred_inv_np
            )
        # for idx in tqdm(range(0, len(prompts), batch_size)):
        #     input_prompts = prompts[idx: idx + batch_size]
        #     gen_image, inverted_code, refine_image, _, inverted_code_image = enhance_model((input_prompts, batch_size), generator=generator)
            
            del enhance_model, pred_inv_noise
        #   Save individual for further evaluate HPSV2
if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)