import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel
import json 
import os 
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np 


class SBV2Gen():
    def __init__(self, path_ckpt_sbv2, model_name = "checkpoints/stable-diffusion-2-1-base"):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        self.unet_gen = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2).to("cuda")
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
    # refine_image = (sbv2_gen.vae.decode(pred_original_sample).sample + 1) / 2
    return pred_original_sample

def compute_inverted_code(gen_image, sbv2_inverse, encoder_hidden_state):
    input_image = gen_image * 2 - 1
    latents = sbv2_inverse.vae.encode(input_image.to(torch.float32)).latent_dist.sample()
    latents = latents * sbv2_inverse.vae.config.scaling_factor
    predict_inverted_code = sbv2_inverse.unet_inverse(latents, sbv2_inverse.mid_timestep, encoder_hidden_state).sample.to(dtype=torch.float32)
    return predict_inverted_code

# @torch.no_grad()
# def sample(
#     pipe,
#     prompt,
#     start_step=0,
#     start_latents=None,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     num_images_per_prompt=1,
#     negative_prompt="",
#     device="cuda",
# ):
#     latents = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, latents=start_latents, device=device, output_type="latent").images
#     return latents

@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=7.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=None,
):

    # Encode prompt
    bsz = len(prompt)
    negative_prompt = [""] * bsz

    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(bsz, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    return latents

@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=None,
):

    negative_prompt = [""] * len(prompt)
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return intermediate_latents

@torch.no_grad()
def main(args):
    ckpt_path = args.pipeline_ckpt
    device = torch.device("cuda")

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.precision == "fp16":
        torch_dtype = torch.float16
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    if args.precision == "fp32":
        torch_dtype = torch.float32
        
    #   Load model    
    model_id = args.pipeline_ckpt
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch_dtype, safety_checker=None).to("cuda")    

    #   Generate image
    generator = torch.Generator("cuda").manual_seed(args.seed)

    list_prompts = json.load(open(args.prompt_path, "r"))[:]
    batch_size = 4
    batch_prompts = [list_prompts[i: i + batch_size] for i in range(0,len(list_prompts), batch_size)]

    if args.generator == "sb":
        sbv2 = SBV2Gen(args.sbv2_ckpt)

    #   [B, T, 4, 64, 64]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for idx, prompts in enumerate(batch_prompts[:4]):
            noise = torch.randn(len(prompts), 4, 64, 64, generator=generator, device="cuda", dtype=torch_dtype)

            # gen_images = sample(
            #     pipe,
            #     prompts, 
            #     start_step=0,
            #     start_latents=noise,
            #     guidance_scale=7.5,
            #     num_inference_steps=50,
            #     num_images_per_prompt=1,
            #     negative_prompt="",
            #     device=pipe.device,
            # )
            gen_images = sample(
                pipe,
                prompts, 
                start_step=0,
                start_latents=noise,
                guidance_scale=7.5,
                num_inference_steps=50,
                num_images_per_prompt=1,
                negative_prompt="",
                do_classifier_free_guidance=True,
                device=pipe.device,
            )

            #   Inversion
            invert_latents = invert(
                    pipe,
                    gen_images,
                    prompts,
                    guidance_scale=1,
                    num_inference_steps=args.inversion_steps,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt="",
                    device=pipe.device,
                )
            
            #   Regenerate image
            new_start_latent = invert_latents[-1]


            input_id = tokenize_captions(prompts, sbv2.tokenizer).to("cuda")
            encoder_hidden_state = sbv2.text_encoder(input_id)[0]
            enhance_image = generate_image(sbv2, new_start_latent, encoder_hidden_state, sbv2.last_timestep) 

            enhance_image_ddim = sample(
                pipe,
                prompts, 
                start_step=0,
                start_latents=new_start_latent,
                guidance_scale=1,
                num_inference_steps=args.inversion_steps,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
                device=pipe.device,
            )
            

            #   Decode image
            decode_latents = torch.cat([gen_images, enhance_image, enhance_image_ddim, new_start_latent])
            images = pipe.decode_latents(decode_latents)                    #   This return a numpy array
            images = torch.from_numpy(images).permute(0,3,1,2)
            
            #   Save results
            save_image(
                make_grid(images, nrow=len(prompts)), f"{args.save_dir}/image_{idx}.png"
            )

if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = OmegaConf.from_cli()
    main(args)