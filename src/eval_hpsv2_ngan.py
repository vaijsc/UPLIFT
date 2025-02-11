#!/usr/bin/env python3
import argparse
import torch
import os
import os.path as osp
import hpsv2
from PIL import Image
import torch.nn as nn
import json
from hpsv2_loss.score import HPSLoss
from accelerate.utils import set_seed
from torchvision import transforms
import re
import sys 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(".")
sys.path.append("..")
def read_tensor_image(image_path:str):
    to_tensor = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    return to_tensor(image)
def natural_key(string):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', string)]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path to json prompt")
    parser.add_argument("--seed", type=int, help="Path to json prompt")
    parser.add_argument("--image_dir", type=str, help="Path to image folder")
    parser.add_argument("--out_dir", type=str, help="Path to image folder")
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    styles = ["anime", "concept_art", "paintings", "photo"]
    with torch.no_grad():
        hps_model = HPSLoss()
        res_save = args.out_dir + f"/{os.path.basename(args.image_dir)}.txt"
        os.makedirs(args.out_dir, exist_ok=True)
        result = ""
        for style in styles:
            image_folder = osp.join(args.image_dir, f"hpsv2_benchmark_{style}")
            if not osp.exists(image_folder):
                continue
            image_names = sorted(os.listdir(image_folder), key=natural_key)
            total_scores = 0.0
            json_file = osp.join(args.json, f"hpsv2_benchmark_{style}.json")
            f = open(json_file, 'r')
            num = len(image_names)
            prompts = (json.load(f))[:num]
            
            for i in tqdm(range(len(prompts))):
                image_path = osp.join(image_folder, image_names[i])
                image = read_tensor_image(image_path)
                hpsv2_score = hps_model(image.unsqueeze(0), [prompts[i]]) 
                total_scores += hpsv2_score
            total_scores = total_scores / len(prompts)
            result += f"{style}_{round(float(total_scores), 3)}\n"
        print(result)
        written_file = open(res_save, 'w')
        written_file.write(result)
        written_file.close()




