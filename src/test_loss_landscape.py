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
import random
import numpy as np

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
    return refine_image, pred_original_sample

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
    def __call__(self, prompts, noise):
        bsz = noise.shape[0]

        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0]
        gen_image, recon_x = generate_image(self.sbv2_gen, noise, encoder_hidden_state, timesteps=self.sbv2_gen.last_timestep)
        
        
        predict_inverted_code = compute_inverted_code(gen_image, self.sbv2_inverse, encoder_hidden_state)
        # inverted_code_image = (self.sbv2_inverse.vae.decode(
        #     predict_inverted_code / self.sbv2_gen.vae.config.scaling_factor
        # ).sample + 1) / 2 

        refine_image, _ = generate_image(self.sbv2_gen, predict_inverted_code, encoder_hidden_state, timesteps=self.sbv2_gen.last_timestep)

        return gen_image, recon_x, predict_inverted_code, refine_image
        

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
    path_sbv2_gen = args.sbv2_path
    path_sbv2_inverse = args.sbv2_inverse_path

    if not osp.isdir(args.save_dir) :
        os.makedirs(f"{args.save_dir}/compare_images")
        os.makedirs(f"{args.save_dir}/enhance_images")
        os.makedirs(f"{args.save_dir}/original_images")
        os.makedirs(f"{args.save_dir}/data")

    loss_fn = torch.nn.MSELoss(reduction="none")
    prompts = get_prompts(args.prompt_path)
    enhance_model = SBV2Enhance(path_sb_v2_gen_model=path_sbv2_gen, path_sb_v2_inverse_model=path_sbv2_inverse)
    batch_size = 1
    count = 0

    random_seed = 8138
    generator = torch.Generator("cuda").manual_seed(random_seed)

    errors = [] 
    errors_enhance = [] 
    save_all = []
    for idx in tqdm(range(0, len(prompts), batch_size)):  
        save_res = {} 
        input_prompts = prompts[idx: idx + batch_size]
        noise = torch.randn(len(input_prompts), 4, 64,64, device="cuda", generator=generator)

        input_id = tokenize_captions(input_prompts, enhance_model.sbv2_gen.tokenizer).to("cuda")
        
        encoder_hidden_state = enhance_model.sbv2_gen.text_encoder(input_id)[0]
        timesteps = enhance_model.sbv2_gen.last_timestep
        gen_image, recon_image, inverted_code, refine_image= enhance_model(input_prompts, noise=noise)

        # save_image(
        #     make_grid(torch.cat([gen_image, refine_image]), nrow=2),
        #     f"{args.save_dir}/compare_images/{idx}.png"
        # )

        eps = torch.randn_like(noise) * 0.01
        pertubed_noise = noise + eps
        enhance_pertubed_noise = inverted_code + eps

        test_images, test_clean_latents = generate_image(
            enhance_model.sbv2_gen, torch.cat([noise, pertubed_noise]), encoder_hidden_state.repeat(2,1,1), timesteps
        )
        
        orig_img, pertubed_img = test_clean_latents.chunk(2)
        error_gaussian = loss_fn(orig_img, pertubed_img).clone().detach().cpu().permute(0,2,3,1).numpy()
        errors.append(error_gaussian)

        #   Save result for gaussian noise (bad latent)
        save_res["gaussian_noise"] = inverted_code.clone().detach().cpu().permute(0,2,3,1).numpy()
        save_res["gaussian_error"] = error_gaussian
        save_res["original_image"] = orig_img.clone().detach().cpu().permute(0,2,3,1).numpy()

        del test_images, test_clean_latents, orig_img, pertubed_img 
        torch.cuda.empty_cache()

        enhance_test_images, enhance_test_clean_latents = generate_image(
            enhance_model.sbv2_gen, torch.cat([inverted_code, enhance_pertubed_noise]), encoder_hidden_state.repeat(2,1,1), timesteps
        )
        enhance_orig_img, enhance_pertubed_img =  enhance_test_clean_latents.chunk(2)
        error_inverted = loss_fn(enhance_orig_img, enhance_pertubed_img).clone().detach().cpu().permute(0,2,3,1).numpy()
        errors_enhance.append(error_inverted)

        save_res["inverted_code"]  = enhance_orig_img.clone().detach().cpu().permute(0,2,3,1).numpy()
        save_res["inverted_error"] = error_inverted
        save_res["enhance_image"] = enhance_orig_img.clone().detach().cpu().permute(0,2,3,1).numpy()

        save_all.append(save_res)
        del enhance_test_images, enhance_test_clean_latents, enhance_orig_img, enhance_pertubed_img 
        torch.cuda.empty_cache()
        
    torch.save(save_all, f"{args.save_dir}/vis_loss.pth")
    breakpoint()


    
    print(f"Gaussian noise: mean: {np.mean(errors)}; std: {np.std(errors)}")
    print(f"Inverted noise: mean: {np.mean(errors_enhance)}; std: {np.std(errors_enhance)}")

if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)