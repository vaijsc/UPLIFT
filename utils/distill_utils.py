from PIL import Image
import torch


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def E_(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_alpha_sigma(scheduler, x, t):
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(x.device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(x.device)
    # Get alpha and sigma of the same size as x to be able to multiply with it 
    alpha = E_(sqrt_alphas_cumprod, t, x.shape)
    sigma = E_(sqrt_one_minus_alphas_cumprod, t, x.shape)
    return alpha, sigma

def make_step(scheduler, pred, timesteps, latent):
    """ Generate the prediced latent and original sample """
    z = torch.ones_like(pred)   # predicted previous latent
    x = torch.ones_like(pred)   # predicted original sample
    for i in range(timesteps.shape[0]):
        step_out = scheduler.step(pred[i], timesteps[i].item(), latent[i])
        z[i], x[i] = step_out.prev_sample, step_out.pred_original_sample
    return (z, x)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h, _ = imgs[0].shape
    grid = Image.new('RGB', size=(cols*w, rows*h))

    images = (imgs * 255).astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    
    for i, img in enumerate(pil_images):
        grid.paste(img, box=((i%cols)*w, (i//cols)*h))

    return grid