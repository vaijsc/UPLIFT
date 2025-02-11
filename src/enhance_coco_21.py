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
import json 
import csv
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
import time
from diffusers import AutoPipelineForText2Image
from lib.swiftbrush import SBV2Gen, SBV2Inverse
# Define transformations outside the loop
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
class SBV2Gen1():
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

class SBV2Inverse1():
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
        self.last_timestep = T
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
        self.sbv2_gen = SBV2Gen1(path_sb_v2_gen_model)
        self.sbv2_inverse = SBV2Inverse1(path_sb_v2_inverse_model)

    @torch.no_grad()
    def __call__(self, gen_image, prompts):
        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0]
        

        # Stage 2.1: Find inverted code
        predict_inverted_code = compute_inverted_code(gen_image, self.sbv2_inverse, encoder_hidden_state)

        # Stage 2.2: Gen img with inverted code
        refine_image = generate_image(self.sbv2_gen, predict_inverted_code, encoder_hidden_state, timesteps=self.sbv2_gen.last_timestep)

        return refine_image

def get_prompts(path):
    _, ext = path.split(".")
    
    if ext == "txt":
        f = open(path, "r")
        objs = f.read() 
        prompts = objs.split("\n")
        return prompts
    
    elif ext == "json":
        return json.load(open(path, "r"))

def read_data(file_path):
    metadata = json.load(open(file_path, 'r'))["labels"]
    return metadata

def main(args):
    path_sbv2_gen = args.sbv2_path
    path_sbv2_inverse = args.sbv2_inverse_path
    start_idx = args.start
    end_idx = args.end
    if not osp.isdir(args.save_dir) :
        os.makedirs(args.save_dir)
    start = time.time()
    # prompts = get_prompts(args.prompt_path)
    if "turbo" in path_sbv2_gen:
        print("using turbo enhance generator ")
        sbgen = SBV2Gen(model_name = "checkpoints/stable-diffusion-2-1-base")
        sb_inverted = SBV2Inverse(path_sbv2_inverse)
        pipe = AutoPipelineForText2Image.from_pretrained("checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c", torch_dtype=torch.float32, variant="fp16")
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        print(pipe.vae.device)
        
    else:
        enhance_model = SBV2Enhance(path_sb_v2_gen_model=path_sbv2_gen, path_sb_v2_inverse_model=path_sbv2_inverse)
    metadata = read_data("data/coco_30k_21.json")[start_idx:end_idx]
    def process_image(image_id, input_prompt, args):
        path = f"{args.image_folder}/{os.path.basename(image_id)}"
        gen_image = transform(Image.open(path)).unsqueeze(0).to("cuda")
        if "turbo" in path_sbv2_gen:
            encoder_hidden_state = sbgen.encode_prompt(input_prompt)
            inverted_code = sb_inverted.compute_inverted_code(gen_image, pipe.vae, encoder_hidden_state)
            enhance_image = pipe(prompt=input_prompt, num_inference_steps=1, guidance_scale=0.0, output_type="pt", latents=inverted_code).images[0]
        else:
            enhance_image = enhance_model(gen_image, [input_prompt])
        save_image(enhance_image, f"{args.save_dir}/{os.path.basename(image_id)}")
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, image[0], image[1], args)
            for image in metadata
        ]
        for future in tqdm(futures):
            future.result()
    print(time.time() - start)
if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)