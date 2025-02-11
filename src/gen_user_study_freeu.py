import sys
sys.path.append(".")
sys.path.append("..")
import argparse
from tqdm import tqdm
import os
import json
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-ckpt",  type=str, help="Path to the pipeline Stable Diffusion")
    parser.add_argument("--save-dir",  type=str, help="Folder to save generated images")  
    parser.add_argument("--precision",  type=str, default="fp16", help="Precicison for generation process") 
    parser.add_argument("--inference-steps",  type=int, default=50, help="Number of generating steps")
    parser.add_argument("--guidance-scale",  type=float, default=7.5, help="Random seed for reproducibility")
    parser.add_argument("--b1",  type=float, default=7.5, help="Random seed for reproducibility")
    parser.add_argument("--b2",  type=float, default=7.5, help="Random seed for reproducibility")
    parser.add_argument("--s1",  type=float, default=7.5, help="Random seed for reproducibility")
    parser.add_argument("--s2",  type=float, default=7.5, help="Random seed for reproducibility")
    return parser.parse_args()

def main(args):
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
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    register_free_upblock2d(pipe, b1=args.b1, b2=args.b2, s1=args.s1, s2=args.s1)
    register_free_crossattn_upblock2d(pipe, b1=args.b1, b2=args.b2, s1=args.s1, s2=args.s1)

    seeds = [4281, 3428, 6051, 2974, 63]
    prompts = [
        # "A young ginger-haired girl",
        # "A beautiful building in a crowded city",
        # "A hiker standing on top of a mountain admiring the view",
        # "A majestic waterfall cascading down a rocky cliff",
        # "A misty morning in a lush green valley",
        # "A painting of a European maiden",
        # "A peaceful meadow filled with wildflowers and butterflies",
        # "A person admiring a rainbow after a rainstorm in a countryside setting",
        # "A person standing under a cherry blossom tree in full bloom",
        # "A person walking on a beach at sunset",
        # "A serene lake surrounded by snow-capped mountains",
        # "A vibrant field of tulips in full bloom",
        # "A woman reading a book under a tree in a garden",
        # "An elderly couple sitting on a park bench in autumn",
        # "Exhibition design, realistic, super detailed",
        # "Planet Earth, watercolor, blue, 8K",
        # "Technical drawing, tall buildings, big city, night",
        # "Two friends having a picnic in a meadow full of wildflowers",
        # "A tranquil bamboo forest with a path leading through the tall stalks",
        # "A child playing in a field of sunflowers"
        "A hairy man lying on a bench besides a bush.",
        "A bunch of people waiting in line by a rail.",
        "A counter in a coffee house with choices of coffee and syrup flavors.",
        "Motorcycles parked on the sidewalk next to a road.",
        "A dresser in a room that is painted bright yellow.",
        "A person with his head out of a window while on a train.",
        "A tiled bathroom with a toilet and sink inside of it.",
        "A man sitting in a chair, in a black and white photo.",
        "A bathroom with clear glass shower door and tile floor.",
        "A dog sitting in a bathroom with a urinal and a torn wall.",
        "A white expensive car parked on top of a cement slab."
    ]
    num_imgs_per_prompt = 6
    noise_folder = f"main_exps/user_study_sd21_2/noise"

    save_folder = f"{args.save_dir}"
    print(f">>> Generate images and save images to {save_folder}")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    for seed in seeds:
        seed_dir = f"{save_folder}/{seed}"
        if not os.path.isdir(seed_dir):
            os.makedirs(seed_dir)
        # generator = torch.Generator("cuda").manual_seed(seed)      
        noise = torch.load(f"{noise_folder}/seed_{seed}.pth", map_location="cpu").to("cuda")

        for prompt in prompts:
            #   Create a folder for each prompt
            save_img_dir = f"{seed_dir}/{prompt}"
            if not os.path.isdir(save_img_dir):
                os.makedirs(save_img_dir)

            batch = [prompt] * num_imgs_per_prompt
            output = pipe(
                        prompt=batch, 
                        latents=noise,
                        guidance_scale=args.guidance_scale, 
                        num_inference_steps=args.inference_steps, 
                        # generator=generator
                        ).images
            for ii, pil in enumerate(output):
                pil.save(f"{save_img_dir}/image_{ii}.png")

if __name__ == '__main__':
    args = parse_args()
    main(args)


     

