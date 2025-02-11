import torch
import os
import os.path as osp
from PIL import Image
from accelerate.utils import set_seed
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf
from lib.swiftbrush import SBV2Inverse, SBV2Gen
from tqdm import tqdm

from lib.utils import read_hps_data, STYLES

def read_tensor_image(image_path:str):
    to_tensor = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    return to_tensor(image)
    



def main(args):
    
    set_seed(args.seed)
    PATHJSON = "data/hpsv2"
    eval_models = args.eval_models.split(',')
    original_image_path = args.original_image_path
    bs = args.batch_size
    all_original_images, all_prompts = read_hps_data(original_image_path, PATHJSON, args.num)
    sbgen = SBV2Gen()
    with torch.no_grad():
        for model in eval_models:
            if "unet_ema" in os.listdir(model):
                model = f'{model}/unet_ema'
            sb_inverted = SBV2Inverse(model)
            model = osp.basename(osp.dirname(model)) + "_" + osp.basename(model).replace("train_inverse_from_fid", "")
            save_folder = osp.join(osp.dirname(original_image_path), f'enhance_{model}')
            os.makedirs(save_folder, exist_ok=True)
            for style in STYLES:
                # create save style folders
                style_save_folder = osp.join(save_folder, f'hpsv2_benchmark_{style}')
                os.makedirs(style_save_folder, exist_ok=True)
                
                # read original images, prompts
                original_images = all_original_images[style]
                prompts = all_prompts[style]
                for i in tqdm(range(0, len(prompts), bs)):
                    prompt = prompts[i : i + bs]
                    original_image = original_images[i : i + bs]
                    enhance_image = sb_inverted.enhance_image(original_image, sbgen, prompt)
                    for ii, img in enumerate(enhance_image):
                        save_image(img.unsqueeze(0), osp.join(style_save_folder, f"image_{ii + i}.png"))
                    del enhance_image
                    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)