import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import os
import json
import torch
from diffusers import DiffusionPipeline, DDPMScheduler
from torchvision.utils import save_image
from tqdm import tqdm
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

    #   Load model    
    with torch.no_grad():
        model_id = args.pipeline_ckpt
        scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        pipe = DiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch_dtype
    )
        pipe = pipe.to("cuda")
        
        json_lists = os.listdir(args.json)

        for json_file in json_lists:
            basename_json = json_file.split(".")[0]
            save_folder = f"{args.save_dir}/{basename_json}"
            print(f">>> Generate from {json_file} and save images to {save_folder}")
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

            args.json_file = f"{args.json}/{json_file}"
            #   Load prompts
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
                batch = prompts[i : i + batch_size]
                image = inference(pipe, encode_prompt, batch, generator, "cuda", torch_dtype)
                for ii, pil in enumerate(image):
                    save_image(pil, f"{save_folder}/image_{ii + i}.png")
                del image
                torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    main(args)


     

