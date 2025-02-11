import torch
import os.path as osp
import os
import argparse
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image, make_grid
import torchvision
from tqdm import tqdm 
import json 
import numpy as np 
import time 
import sys 
import safetensors
sys.path.append(".")
sys.path.append("..")
from train_scripts.model_baselines import MLP2Layers, UNet

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
    def __init__(self, model_type, dtype="fp32"):
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        if model_type == 'unet' :
            self.inverted_model = UNet().to("cuda")
        elif model_type == "mlp":
            self.inverted_model = MLP2Layers().to("cuda")

    def _init_weight(self, state_dict):
        self.inverted_model.load_state_dict(state_dict, strict=True)

    def __call__(self, gt_x, gt_z):
        return self.inverted_model(gt_x, gt_z)

def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

@torch.inference_mode()
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
    return refine_image, pred_original_sample

def compute_inverted_code(gen_image, sbv2_inverse, encoder_hidden_state):
    input_image = gen_image * 2 - 1
    latents = sbv2_inverse.vae.encode(input_image.to(torch.float32)).latent_dist.sample()
    latents = latents * sbv2_inverse.vae.config.scaling_factor
    predict_inverted_code = sbv2_inverse.unet_inverse(latents, sbv2_inverse.mid_timestep, encoder_hidden_state).sample.to(dtype=torch.float32)
    return predict_inverted_code

class SBV2Enhance():
    def __init__(self, path_sb_v2_gen_model, model_type="unet"):
        # define sb gen and inverse model
        self.sbv2_gen = SBV2Gen(path_sb_v2_gen_model)
        self.sbv2_inverse = SBV2Inverse(model_type=model_type)

    @torch.no_grad()
    def __call__(self, gt_z, prompts):
        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0]
        gen_image, gt_x = generate_image(self.sbv2_gen, gt_z, encoder_hidden_state, self.sbv2_gen.last_timestep)
        # Stage 2.1: Find inverted code
        predict_inverted_code = self.sbv2_inverse(gt_x, gt_z)

        # Stage 2.2: Gen img with inverted code
        refine_image, _ = generate_image(self.sbv2_gen, predict_inverted_code, encoder_hidden_state, timesteps=self.sbv2_gen.last_timestep)

        return gen_image, gt_x, refine_image, predict_inverted_code

def get_prompts(path):
    _, ext = path.split(".")
    
    if ext == "txt":
        f = open(path, "r")
        objs = f.read()
        prompts = objs.split("\n")
        return prompts
    
    elif ext == "json":
        return json.load(open(path, "r"))

@torch.inference_mode()
def main(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
 
    prompts = get_prompts(args.prompt_path)
    path_sbv2_gen = args.sbv2_path
    path_inversion = args.inverse_path

    state_dict = safetensors.torch.load_file(path_inversion)

    enhance_model = SBV2Enhance(path_sbv2_gen, model_type=args.model_type)
    enhance_model.sbv2_inverse._init_weight(state_dict)

    #   Load weight for inverted model
    

    for idx, prompt in tqdm(enumerate(prompts)):
        #   Gen and enhance
        gt_z = torch.randn(1,4,64,64).to("cuda")
        gen_image, gt_x, enhance_image, inverted_code = enhance_model(gt_z, [prompt])

        save_image(
            make_grid(
                torch.cat([gen_image, enhance_image]), nrow=1
            ),
            f"{args.save_dir}/image_{idx}.png"
        )
        # breakpoint()

if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)