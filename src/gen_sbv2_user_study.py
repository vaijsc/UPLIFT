import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import os
import json
import torch
from diffusers import DiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from torchvision.utils import save_image

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-ckpt",  type=str, help="Path to the pipeline Stable Diffusion")
    parser.add_argument("--save-dir",  type=str, help="Folder to save generated images")  
    parser.add_argument("--precision",  type=str, default="fp16", help="Precicison for generation process") 
    return parser.parse_args()

@torch.no_grad()
def encode_prompt(pipe, prompt):
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    with torch.no_grad():
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


@torch.no_grad()
def inference(pipe, encode_func, prompt, generator, device, weight_dtype):
    vae = pipe.vae

    prompt_embed = encode_func(pipe, prompt)["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)

    input_shape = (prompt_embed.shape[0], 4, 64, 64)
    input_noise = torch.randn(input_shape, generator=generator, device=device, dtype=weight_dtype)

    pred_original_sample = predict_original(pipe, input_noise, prompt_embed)
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=weight_dtype)).sample.float()
    return (image + 1) / 2


def predict_original(pipe, input_noise, prompt_embeds):
    unet = pipe.unet
    scheduler = pipe.scheduler

    max_timesteps = torch.ones((input_noise.shape[0],), dtype=torch.int64, device=input_noise.device)
    max_timesteps = max_timesteps * (scheduler.config.num_train_timesteps - 1)

    alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047) ** 0.5
    model_pred = unet(input_noise, max_timesteps, prompt_embeds).sample

    latents = (input_noise - sigma_T * model_pred) / alpha_T
    return latents
def main(args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.precision == "fp16":
        torch_dtype = torch.float16
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    if args.precision == "fp32":
        torch_dtype = torch.float32

    seeds = [4281, 3428, 6051, 2974, 63]
    prompts = [
        "A young ginger-haired girl",
        "A beautiful building in a crowded city",
        "A hiker standing on top of a mountain admiring the view",
        "A majestic waterfall cascading down a rocky cliff",
        "A misty morning in a lush green valley",
        "A painting of a European maiden",
        "A peaceful meadow filled with wildflowers and butterflies",
        "A person admiring a rainbow after a rainstorm in a countryside setting",
        "A person standing under a cherry blossom tree in full bloom",
        "A person walking on a beach at sunset",
        "A serene lake surrounded by snow-capped mountains",
        "A vibrant field of tulips in full bloom",
        "A woman reading a book under a tree in a garden",
        "An elderly couple sitting on a park bench in autumn",
        "Exhibition design, realistic, super detailed",
        "Planet Earth, watercolor, blue, 8K",
        "Technical drawing, tall buildings, big city, night",
        "Two friends having a picnic in a meadow full of wildflowers",
        "A tranquil bamboo forest with a path leading through the tall stalks",
        "A child playing in a field of sunflowers"
    ]
    num_imgs_per_prompt = 6

    save_folder = f"{args.save_dir}"
    print(f">>> Generate images and save images to {save_folder}")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    #   Load model    
    with torch.no_grad():
        model_id = args.pipeline_ckpt
        scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        pipe = DiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch_dtype
    )
        pipe.unet = UNet2DConditionModel.from_pretrained("checkpoints/sbv2_fid81", subfolder="unet").to("cuda",
        dtype=torch_dtype
    )

        pipe = pipe.to("cuda") 
        
        for seed in seeds:
            seed_dir = f"{save_folder}/{seed}"
            if not os.path.isdir(seed_dir):
                os.makedirs(seed_dir)
            generator = torch.Generator("cuda").manual_seed(seed)      
            for prompt in prompts:
                #   Create a folder for each prompt
                save_img_dir = f"{seed_dir}/{prompt}"
                if not os.path.isdir(save_img_dir):
                    os.makedirs(save_img_dir)

                batch = [prompt] * num_imgs_per_prompt
                image = inference(pipe, encode_prompt, batch, generator, "cuda", torch_dtype)
                for ii, pil in enumerate(image):
                    save_image(pil, f"{save_img_dir}/image_{ii}.png")
                del image
                torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    main(args)


     

