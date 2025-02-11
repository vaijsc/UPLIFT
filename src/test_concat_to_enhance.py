import torch
import os.path as osp
import os
import argparse
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image, make_grid
from tqdm import tqdm 
import json 

class SBV2Gen():
    def __init__(self, path_ckpt_sbv2, model_name = "checkpoints/stable-diffusion-2-1-base"):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        self.unet_gen = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to("cuda")
        self.unet_gen.eval()

        self.last_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        
        # prepare stuff
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
        
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

def generate_image(sbv2_gen, noise, encoder_hidden_state, return_latents):
    model_pred = sbv2_gen.unet_gen(noise, sbv2_gen.last_timestep, encoder_hidden_state).sample
    pred_original_sample = (noise - sbv2_gen.sigma_t * model_pred) / sbv2_gen.alpha_t
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

    if not return_latents:
        refine_image = (sbv2_gen.vae.decode(pred_original_sample).sample + 1) / 2
        return refine_image
    return pred_original_sample

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
    def __call__(self, prompts, generator):
        total_image = len(prompts)
        noise = torch.randn(total_image, 4, 64, 64, device="cuda", generator=generator)
        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0]
        gen_image = generate_image(self.sbv2_gen, noise, encoder_hidden_state)
        ## Stage 2: Inverse image to latent code and gen again with SBV2
        
        # Stage 2.1: Find inverted code
        predict_inverted_code = compute_inverted_code(gen_image, self.sbv2_inverse, encoder_hidden_state)
        inverted_code_image = (self.sbv2_inverse.vae.decode(predict_inverted_code).sample + 1) / 2 

        # Stage 2.2: Gen img with inverted code
        refine_image = generate_image(self.sbv2_gen, predict_inverted_code, encoder_hidden_state)
        return gen_image, predict_inverted_code, refine_image, noise, inverted_code_image

class SBv2EnhanceConcat(SBV2Enhance):
    def __init__(self, path_sbv2_gen, path_sbv2_inversion):
        self.sbv2_gen = SBV2Gen(path_sbv2_gen)
        self.sbv2_inverse = SBV2Inverse(path_sbv2_inversion)
 
    def create_new_features(self, x1, x2, replace_ids=None):
        #   x1: [B, C1, h, w]
        #   x2: [B, C2, h, w]
        # if max(replace_ids) >= x1.shape[1]:
        #     raise ValueError()
        weight = 1.
        idx = 0
        new_x = x2.clone()        
        new_x[:, idx, :, :] =   weight * x1[:, idx, :, :] + (1. - weight) * x2[:, idx, :, :]
        
        return new_x

    @torch.no_grad()
    def __call__(self, prompts, generator):
        use_old_noise =  hasattr(self, "use_old_noise")
        total_image = len(prompts)

        noise = torch.randn(total_image, 4, 64, 64, device="cuda", generator=generator)
        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0]

        #   Clean latents
        gen_image = generate_image(self.sbv2_gen, noise, encoder_hidden_state, return_latents=True) * self.sbv2_gen.vae.config.scaling_factor
        if use_old_noise:
            enhance_latents = self.create_new_features(x1=gen_image, x2=noise)
        else:
            new_noise = torch.randn(gen_image.shape, device=gen_image.device)
            noise = new_noise
            enhance_latents = self.create_new_features(x1=gen_image, x2=noise)

        inverted_code_image =  (self.sbv2_gen.vae.decode(
            new_noise / self.sbv2_gen.vae.config.scaling_factor
        ).sample + 1) / 2

        model_pred = self.sbv2_gen.unet_gen(enhance_latents, self.sbv2_gen.last_timestep, encoder_hidden_state).sample
        pred_original_sample = (enhance_latents - self.sbv2_gen.sigma_t * model_pred) / self.sbv2_gen.alpha_t

        pred_original_sample = pred_original_sample / self.sbv2_gen.vae.config.scaling_factor

        refine_image = (self.sbv2_gen.vae.decode(pred_original_sample).sample + 1) / 2
        gen_images = (self.sbv2_gen.vae.decode(
            gen_image / self.sbv2_gen.vae.config.scaling_factor
        ).sample + 1) / 2

        return gen_images, None, refine_image, noise, inverted_code_image
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
    path_sbv2_gen = args.sbv2_path
    path_sbv2_inverse = args.sbv2_inverse_path

    if not osp.isdir(args.save_dir) :
        os.makedirs(f"{args.save_dir}/compare_images")
        os.makedirs(f"{args.save_dir}/enhance_images")
        os.makedirs(f"{args.save_dir}/original_images")

    prompts = get_prompts(args.prompt_path)
    enhance_model = SBv2EnhanceConcat(path_sbv2_gen=path_sbv2_gen, path_sbv2_inversion=path_sbv2_inverse)
    batch_size = args.bsz
    count = 0
    # generator = torch.Generator("cuda").manual_seed(6969)

    for idx in tqdm(range(0, len(prompts), batch_size)):
        input_prompts = prompts[idx: idx + batch_size]
        gen_image, inverted_code, refine_image, _, inverted_code_image = enhance_model(input_prompts, generator=None)
        

        #   Save individual for further evaluate HPSV2
        for i, image in enumerate(refine_image):
            basename = "_".join(input_prompts[i][:100].replace(".", "").replace("/", "_").split(" "))
            save_image(image.unsqueeze(0), f"{args.save_dir}/enhance_images/{i+idx}_{basename}.png") 

        #   Save qualitatives
        if count * batch_size < 16:

            vis_images = torch.cat([gen_image, refine_image, inverted_code_image])
            save_image(
                make_grid(vis_images, nrow=batch_size), f"{args.save_dir}/compare_images/{count}.png"
            )    
            count += 1
        else:
            exit()
if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)