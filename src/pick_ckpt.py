from lib.swiftbrush import SBV2Gen, SBV2Inverse
from omegaconf import OmegaConf
from tqdm import tqdm
import os.path as osp
import os
from torchvision.utils import save_image
from accelerate.utils import set_seed
import torch
set_seed(0)
def main(args):
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    sbgen = SBV2Gen()
    encoder_hidden_state = sbgen.encode_prompt(8)
    image = sbgen.generate_image(encoder_hidden_state)
    save_image(image, osp.join(output_path, "gen.jpg"))
    ckpt_folder = args.training_output
    folders = sorted(os.listdir(ckpt_folder))
    for folder in tqdm(folders):
        if "checkpoint" in folder:
            ckpt_path = osp.join(ckpt_folder, folder)
            ckpt_step = folder[11:]
            sb_inverted = SBV2Inverse(ckpt_path)
            enhance, inverted_code = sb_inverted.enhance_image(image, sbgen, encoder_hidden_state, return_code=True)
            save_image(torch.cat((sbgen.decode_image(inverted_code), enhance), dim=0), osp.join(output_path, f"enhance_{ckpt_step}.jpg"))
            
if __name__ == '__main__':
    args = OmegaConf.from_cli()
    main(args)