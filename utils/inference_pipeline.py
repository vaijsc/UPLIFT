# ------------------------------------------------------------------------------------
# Copyright (c) 2023 Nota Inc. All Rights Reserved.
# Code modified from https://huggingface.co/blog/stable_diffusion
# ------------------------------------------------------------------------------------

import diffusers
import torch
import gc
import json
from peft import LoraModel, LoraConfig, set_peft_model_state_dict

diffusers_version = int(diffusers.__version__.split('.')[1])

class InferencePipeline:
    def __init__(self, weight_folder, seed, device):
        self.weight_folder = weight_folder
        self.seed = seed
        self.device = torch.device(device)

        self.pipe = None
        self.generator = None

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self, selective_guidance=False): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        if selective_guidance:
            from .selective_guidance import StableDiffusionPipeline
        else: 
            from diffusers import StableDiffusionPipeline
        if diffusers_version == 15: # for the specified version in requirements.txt
            if self.weight_folder in ["Lykon/DreamShaper", "SimianLuo/LCM_Dreamshaper_v7"]:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.weight_folder,
                    device_map=None, low_cpu_mem_usage=False,
                    torch_dtype=torch.float16,
                ).to(self.device)
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                    torch_dtype=torch.float16).to(self.device)
            self.pipe.safety_checker = lambda images, clip_input: (images, False) 
        elif diffusers_version >= 19: # for recent diffusers versions
            if self.weight_folder in ["Lykon/DreamShaper", "SimianLuo/LCM_Dreamshaper_v7"]:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.weight_folder,
                    device_map=None, low_cpu_mem_usage=False,
                    safety_checker=None,
                    torch_dtype=torch.float16,
                ).to(self.device)
            elif self.weight_folder == "Deci/DeciDiffusion-v2-0":
                self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder, custom_pipeline=self.weight_folder, torch_dtype=torch.float16, safety_checker=None)
                self.pipe.unet = self.pipe.unet.from_pretrained(self.weight_folder, subfolder='flexible_unet', torch_dtype=torch.float16)
                self.pipe = self.pipe.to(self.device)
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                safety_checker=None, torch_dtype=torch.float16).to(self.device)
        else: # for the versions between 0.15 and 0.19, the benchmark scores are not guaranteed.
            raise Exception(f"Use diffusers version as either ==0.15.0 or >=0.19 (from current {diffusers.__version__})")

        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        # self.pipe.unet = self.pipe.unet.half().to(self.device)

    def generate(self, prompt: str, n_steps: int, img_sz: int, guidance_scale: float = 7.5, opt_percentage=None):
        if opt_percentage:
            out = self.pipe(
                prompt,
                num_inference_steps=n_steps,
                height = img_sz,
                width = img_sz,
                generator=self.generator,
                guidance_scale=guidance_scale,
                opt_percentage=opt_percentage
            ) 
        else:
            out = self.pipe(
                prompt,
                num_inference_steps=n_steps,
                height = img_sz,
                width = img_sz,
                generator=self.generator,
                guidance_scale=guidance_scale,
            ) 
        return out.images[0]
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_sdm_params(self):
        params_unet = self._count_params(self.pipe.unet)
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        params_total = params_unet + params_text_enc + params_image_dec
        return f"Total {(params_total/1e6):.1f}M (U-Net {(params_unet/1e6):.1f}M; TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"


def load_and_set_lora_ckpt(pipe, weight_path, config_path, dtype):
    device = pipe.unet.device

    with open(config_path, "r") as f:
        lora_config = json.load(f)
    lora_checkpoint_sd = torch.load(weight_path, map_location=device)
    unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
    text_encoder_lora_ds = {
        k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
    }

    unet_config = LoraConfig(**lora_config["peft_config"])
    pipe.unet = LoraModel(unet_config, pipe.unet)
    set_peft_model_state_dict(pipe.unet, unet_lora_ds)

    if "text_encoder_peft_config" in lora_config:
        text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
        pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe