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
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionSAGPipeline
from accelerate.utils import set_seed
import time

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-ckpt",  type=str, help="Path to the pipeline Stable Diffusion")
    parser.add_argument("--save-dir",  type=str, help="Folder to save generated images")  
    parser.add_argument("--precision",  type=str, default="fp16", help="Precicison for generation process") 
    parser.add_argument("--json",  type=str, help="Path to json file containing prompts")
    parser.add_argument("--seed",  type=int, help="Random seed for reproducibility")
    parser.add_argument("--inference-steps",  type=int, default=50, help="Number of generating steps")
    parser.add_argument("--guidance-scale",  type=float, default=7.5, help="Random seed for reproducibility")
    parser.add_argument("--start",  type=int, help="Random seed for reproducibility")
    parser.add_argument("--end",  type=int, default=-1, help="Random seed for reproducibility")
    parser.add_argument("--batch-size",  type=int, help="Random seed for reproducibility")
    parser.add_argument("--turbo",  type=bool, default=False, help="Random stuff for formalizing code")
    parser.add_argument("--sdxl",  action="store_true")
    parser.add_argument("--teacher-lora",  action="store_true")
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

    if args.turbo:
        assert args.inference_steps == 1
        assert args.guidance_scale == 0
        
    #   Load model    
    model_id = args.pipeline_ckpt
    pipe = StableDiffusionSAGPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    
    set_seed(args.seed)

    save_folder = f"{args.save_dir}"
    print(f">>> Generate from {args.json} and save images to {save_folder}")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    
    #   Load prompts
    data = json.load(open(
        args.json, mode="r"
    ))
    #   Set seed
    batch_size = args.batch_size
    start_index = max(0, args.start)
    if args.end == -1:
        end_index = len(data)
    else:
        end_index = min(len(data), args.end)
    generator = torch.Generator("cuda").manual_seed(args.seed)      
    total_time = 0.0

    for i in tqdm(range(start_index, end_index, batch_size)):
        batch = data[i : i + batch_size]
        filenames = list(map(lambda x: x["filename"], batch))
        prompts = list(map(lambda x: x["prompt"], batch))
        start_time = time.time()

        output = pipe(prompts, 
                    guidance_scale=args.guidance_scale, 
                    num_inference_steps=args.inference_steps, 
                    generator=generator,
                    sag_scale=1.0).images

        # for ii, pil in enumerate(output):
        #     pil.save(f"{save_folder}/{filenames[ii]}")
        end_time = time.time()
        total_time += (end_time - start_time)
    print(total_time)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)


     

