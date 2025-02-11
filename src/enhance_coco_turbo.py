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

def read_data(file_path):
    metadata = json.load(open(file_path, 'r'))["labels"]
    return metadata

def main(args):
    path_sbv2_inverse = args.sbv2_inverse_path
    start_idx = args.start
    end_idx = args.end
    if not osp.isdir(args.save_dir) :
        os.makedirs(args.save_dir)
    start = time.time()
    print("using turbo enhance generator ")
    sbgen = SBV2Gen(model_name = "checkpoints/stable-diffusion-2-1-base")
    sb_inverted = SBV2Inverse(path_sbv2_inverse)
    pipe = AutoPipelineForText2Image.from_pretrained("checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c", torch_dtype=torch.float32, variant="fp16")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(pipe.vae.device)
        
    metadata = read_data("data/coco_30k_21.json")[start_idx:end_idx]
    # def process_image(image_id, input_prompt, args):
    #     path = f"{args.image_folder}/{os.path.basename(image_id)}"
    #     gen_image = transform(Image.open(path)).unsqueeze(0).to("cuda")
    #     encoder_hidden_state = sbgen.encode_prompt(input_prompt)
    #     inverted_code = sb_inverted.compute_inverted_code(gen_image, pipe.vae, encoder_hidden_state)
    #     enhance_image = pipe(prompt=input_prompt, num_inference_steps=1, guidance_scale=0.0, output_type="pt", latents=inverted_code).images[0]
    #     save_image(enhance_image, f"{args.save_dir}/{os.path.basename(image_id)}")
    # # Use ThreadPoolExecutor for parallel processing
    # with ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(process_image, image[0], image[1], args)
    #         for image in metadata
    #     ]
    #     for future in tqdm(futures):
    #         future.result()
    for image in tqdm(metadata):
        path = f"{args.image_folder}/{os.path.basename(image[0])}"
        gen_image = transform(Image.open(path)).unsqueeze(0).to("cuda")
        encoder_hidden_state = sbgen.encode_prompt(image[1])
        inverted_code = sb_inverted.compute_inverted_code(gen_image, pipe.vae, encoder_hidden_state)
        enhance_image = pipe(prompt=image[1], num_inference_steps=1, guidance_scale=0.0, output_type="pt", latents=inverted_code).images[0]
        save_image(enhance_image, f"{args.save_dir}/{os.path.basename(image[0])}")
    print(time.time() - start)
if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)