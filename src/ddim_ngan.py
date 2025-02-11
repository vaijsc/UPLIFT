import torch
import os
import os.path as osp
from lib.swiftbrush import SBV2Gen, SBV2Inverse
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, DDIMScheduler
## Inversion

# describe: I + noise (vary) -> z -> DDIM inversion -> z' -> z' bar xem no co spatial ko
sbgen = SBV2Gen()
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# sbgen.noise_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# sb.noise_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    negative_prompt="",
    device="cuda",
):
    latents = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, latents=start_latents, device=device, output_type="latent").images
    return latents
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=1.0,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
):
    do_classifier_free_guidance = guidance_scale > 1.0
    text_embeddings = sbgen.encode_prompt(prompt)
    uncond_text_embeddings = sbgen.encode_prompt([negative_prompt]*text_embeddings.shape[0])
    text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings])  if do_classifier_free_guidance else text_embeddings
    # Latents are now the specified start latents
    latents = start_latents.clone()
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in range(1, num_inference_steps):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred
    return latents
import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image pairs from two folders.")
    parser.add_argument("chunk_idx", type=int, help="Path to the folder containing original images.")
    args = parser.parse_args()
    SAVEPATH = "_output_ddim13"
    os.makedirs(SAVEPATH, exist_ok=True)
    with torch.no_grad():
        guidance_scale=1.0
        sample_step = 30
        num_inference_steps = 30
        prompts = json.load(open("data/sample.json", 'r'))
        # prompts = ["An Art Deco-inspired ballroom filled with people dressed in 1920s attire, with chandeliers, golden accents, and a jazz band playing on stage.",
        #     "A peaceful Japanese zen garden at sunrise, with a small wooden bridge, koi fish swimming in the pond, and cherry blossom petals floating in the air.",
        #     "A dreamy beach at sunset with waves softly crashing onto a pink-tinted shore, seashells scattered on the sand, a lone palm tree silhouetted against a pastel sky, and a sailboat drifting toward the horizon.",
        #     "A detailed, photorealistic portrait of an elderly woman with intricate facial tattoos, traditional jewelry, and an intense, wise gaze.",
        #     "A close-up of a steampunk-inspired clock with intricate gears, brass elements, and small bursts of steam, set against a vintage Victorian background.",
        #     "An old European village square at night, with cozy cafes and shops lit by warm street lamps, cobblestone paths, people enjoying outdoor dining, and a distant view of a medieval castle on a hill."]
        for idx, prompt in tqdm(enumerate(prompts)):
            clean_prompt = prompt.replace(" ", "_")
            PROMPT_SAVEPATH = f'{SAVEPATH}/{clean_prompt}'
            os.makedirs(PROMPT_SAVEPATH, exist_ok=True)
            noise = sbgen.get_noise(1)
            gen_latent = sample(prompt, start_latents=noise, guidance_scale=guidance_scale, num_inference_steps=sample_step)
            encoder_hidden_state = sbgen.encode_prompt(prompt)
            num_images_per_prompt=1
            inverted_noise = invert(gen_latent, [prompt], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            enhance_multi = sample(prompt, start_latents=inverted_noise, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
            enhance_sb = sbgen.generate_latent(encoder_hidden_state, inverted_noise)
            res_image = torch.cat((gen_latent, enhance_multi, enhance_sb), dim=0)
            save_image(sbgen.decode_image(res_image), os.path.join(PROMPT_SAVEPATH, f"img.jpg"), nrow=3)
