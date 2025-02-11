#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import torch
# from utils.picksocre import *
import hpsv2
from tqdm import tqdm
import sys 
sys.path.append(".")
sys.path.append("..")
from hpsv2_loss.score import HPSLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path to json prompt")
    parser.add_argument("--score", type=str, help="Path to json prompt")
    parser.add_argument("--image-dir", type=str, help="Path to image folder")
    parser.add_argument("--save-res-dir", type=str, help="Path to save results")
    return parser.parse_args()

def get_prompt_name(folder_image):
    ANIME="hpsv2_benchmark_anime.json"
    CONCEPT="hpsv2_benchmark_concept_art.json"
    PAINTINGS="hpsv2_benchmark_paintings.json"
    PHOTO="hpsv2_benchmark_photo.json"
    PARTIPROMPT="partiprompt.json"

    if "anime" in folder_image:
        return ANIME

    elif "photo" in folder_image:
        return PHOTO

    elif "paintings" in folder_image:
        return PAINTINGS
    
    elif "concept" in folder_image:
        return CONCEPT


def eval_hpsv2(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    if not os.path.isdir(args.save_res_dir):
        os.makedirs(args.save_res_dir) 

    import DiffusionDPO.utils.hps_utils as hps
    score_fn = hps.Selector(device="cuda")
    # score_fn = HPSLoss(ckpt="checkpoints/HPS_v2_compressed.pt", device="cuda")
    #   Get subfolder images
    n_folders_image = sorted(os.listdir(args.image_dir)) 
    print(f">>> Found {len(n_folders_image)} folders in {args.image_dir}")  

    for folder in n_folders_image:
        if folder == "hpsv2_results":
            continue
        n_images = len(os.listdir(f"{args.image_dir}/{folder}"))
        print(f">>> Found {n_images} images in {folder}")

        prompt_name = get_prompt_name(folder)
        args.json_file = f"{args.json}/{prompt_name}"
        prompts = json.load(open(
            args.json_file, mode="r"
        ))
        res = []
        scores = []
        for idx, prompt in tqdm(enumerate(prompts)):
            image_dir = f'{args.image_dir}/{folder}'
            image_path = f"{image_dir}/image_{idx}.png"
            # score = hpsv2.score(image_path, prompt, hps_version="v2.1")[0]
            score = score_fn.score(image_path, prompt)[0]
            res.append({
                "prompt": prompt,
                "image_path": image_path,
                "human_preference_score": score.item()
            })
            scores.append(score)
        res.append({
            "mean_score": np.mean(scores).item()
        })
        mean_hpsv2 = np.mean(scores).item() 
        basename = image_dir.split("/")[-1]
        print(f"Eval on {basename}\nMean hpsv2: {mean_hpsv2} over {len(prompts)} images")
        print("=========================================================================================")
        json.dump(
            res,
            open(f"{args.save_res_dir}/eval_{basename}.json", mode="w")
        )
if __name__ == '__main__':
    args = parse_args()
    if args.score == "pickscore":
        eval_pickscore(args)
    elif args.score == "hpsv2":
        eval_hpsv2(args)
