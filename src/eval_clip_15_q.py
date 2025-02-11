# ------------------------------------------------------------------------------------
# Copyright (c) 2023 Nota Inc. All Rights Reserved.
# Code modified from https://github.com/mlfoundations/open_clip/tree/37b729bc69068daa7e860fb7dbcf1ef1d03a4185#usage
# ------------------------------------------------------------------------------------
import sys 
sys.path.append(".")
sys.path.append("..")
import os
import argparse
import torch
import open_clip
from PIL import Image
from utils.misc import get_file_list_from_csv
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_txt", type=str, default="./results/bk-sdm-small/im256_clip.txt")
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")   
    parser.add_argument("--img_dir", type=str, default="./results/bk-sdm-small/im256")  
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14',
    #                                                              pretrained='laion2b_s34b_b88k',
    #                                                              device=args.device)

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14',
                                                                 pretrained='laion2b_s12b_b42k',
                                                                 device="cuda")
    model.float()
    # tokenizer = open_clip.get_tokenizer('ViT-g-14')

    _, file_extension = os.path.splitext(args.data_list)
    if file_extension.lower() == '.csv':
        file_list = get_file_list_from_csv(args.data_list)
    elif file_extension.lower() == '.txt':
        with open(args.data_list, 'r', encoding='utf-8') as file:
            file_list = file.readlines()
    elif file_extension.lower() == '.json':
        import json
        with open(args.data_list, "r") as f:
            data = json.load(f)
            # file_list = data["labels"]
    else:
        raise ValueError(f"Does not support this file type {file_extension} for data_list")    

    # file_list = file_list[:30000]
    # length = len(file_list)
    score_arr = []
    mean_score = 0
    for item in data:
        img_path = os.path.join(args.img_dir, item["filename"])
        val_prompt = "A photo depicts " + item["prompt"]

    # for i, file_info in enumerate(tqdm(file_list)):
    #     if file_extension.lower() == '.csv':
    #         # img_path = os.path.join(args.img_dir, file_info[0].replace("jpg", "png"))
    #         img_path = os.path.join(args.img_dir, file_info[0])
    #         val_prompt = file_info[1]
    #         val_prompt = "A photo depicts " + val_prompt
    #     elif file_extension.lower() == '.txt': 
    #         val_prompt = file_info
    #         normed_prompt = val_prompt.replace(',', '').replace('\n', '').replace(" ", "_").replace('/', '_')[:250]
    #         img_path = os.path.join(args.img_dir, f"{normed_prompt}.jpg")
    #     elif file_extension.lower() == '.json':
    #         img_name = file_info[0]
    #         img_name = img_name.split("/")[-1]
    #         img_path = os.path.join(args.img_dir, img_name)
    #         val_prompt = file_info[1]
    #         val_prompt = "A photo depicts " + val_prompt
       
        text = open_clip.tokenize([val_prompt]).to(args.device)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            probs = F.cosine_similarity(image_features, text_features, dim=1)
            mean_score += probs[0]

    mean_score /= len(data)
    # print(length)
    print(mean_score) 
        # print(f"{i}/{length} | {val_prompt} | probs {probs[0][0]}") 
    
    with open(args.save_txt, 'a+') as f:
        f.write(f"{args.img_dir}: clip score {mean_score}\n")