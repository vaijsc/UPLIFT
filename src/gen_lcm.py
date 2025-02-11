from diffusers import DiffusionPipeline
import torch
import json 
from PIL import Image 
import numpy as np 
import argparse
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir",  type=str, help="Folder to save generated images")  
    parser.add_argument("--precision",  type=str, default="fp16", help="Precicison for generation process") 
    parser.add_argument("--json",  type=str, help="Path to json file containing prompts")
    parser.add_argument("--seed",  type=int, help="Random seed for reproducibility")
    parser.add_argument("--start",  type=int, help="Random seed for reproducibility")
    parser.add_argument("--end",  type=int, default=-1, help="Random seed for reproducibility")
    parser.add_argument("--inference-steps",  type=int, help="Random seed for reproducibility")
    parser.add_argument("--batch-size",  type=int, help="Random seed for reproducibility")
    return parser.parse_args()

def main(args):

    # Load scheduler and models
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.precision == "fp16":
        torch_dtype = torch.float16
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    if args.precision == "fp32":
        torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained("checkpoints/models--SimianLuo--LCM_Dreamshaper_v7/snapshots/a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5", torch_dtype=torch_dtype, safety_checker=None).to("cuda")


    for json_file in os.listdir(args.json): 
        args.json_file = f"{args.json}/{json_file}"
        basename = json_file.split(".")[0]
        args.save_dir_images = f"{args.save_dir}/{basename}"
        print(f">>> Generate from {args.json_file}")
        print(f">>> Save images in {args.save_dir_images}")
        if not os.path.isdir(args.save_dir_images):
            os.makedirs(args.save_dir_images)

        prompts = json.load(open(
            args.json_file, mode="r"
        ))

        #   Set seed
        batch_size = args.batch_size
        start_index = max(0, args.start)
        if args.end == -1:
            end_index = len(prompts)
        else:
            end_index = min(len(prompts), args.end)
        generator = torch.Generator("cuda").manual_seed(args.seed)      
        
        for i in tqdm(range(start_index, end_index, batch_size)):
            batch_prompts = prompts[i : i + batch_size]
            images = pipe(prompt=batch_prompts, 
                            generator=generator,
                            num_inference_steps=args.inference_steps, 
                            guidance_scale=7.5, lcm_origin_steps=50, output_type="pil").images
            
            for ii, img_uint8 in enumerate(images):
                img_uint8.save(f"{args.save_dir_images}/image_{ii + i}.png")
            # breakpoint()
        
if __name__ == '__main__':
    args=  parse_args()
    main(args)