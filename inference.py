import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from blora_utils import BLOCKS, filter_lora, scale_lora, load_lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="B-LoRA inference (modern diffusers / PEFT backend).")
    parser.add_argument(
        "--prompt", type=str, required=True, help="B-LoRA prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--content_B_LoRA", type=str, default=None, help="path for the content B-LoRA"
    )
    parser.add_argument(
        "--style_B_LoRA", type=str, default=None, help="path for the style B-LoRA"
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1., help="alpha parameter to scale the content B-LoRA weights"
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1., help="alpha parameter to scale the style B-LoRA weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=5.0, help="classifier-free guidance scale"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None, help="negative prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducible generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    # Get Content B-LoRA SD
    if args.content_B_LoRA is not None:
        content_B_LoRA_sd = load_lora_state_dict(args.content_B_LoRA)
        content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS['content'])
        content_B_LoRA = scale_lora(content_B_LoRA, args.content_alpha)
    else:
        content_B_LoRA = {}

    # Get Style B-LoRA SD
    if args.style_B_LoRA is not None:
        style_B_LoRA_sd = load_lora_state_dict(args.style_B_LoRA)
        style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
        style_B_LoRA = scale_lora(style_B_LoRA, args.style_alpha)
    else:
        style_B_LoRA = {}

    # Merge B-LoRAs SD
    res_lora = {**content_B_LoRA, **style_B_LoRA}

    # Load into the UNet: `load_lora_weights` accepts a state dict directly and
    # converts diffusers-format keys to the PEFT layout internally.
    pipeline.load_lora_weights(res_lora, adapter_name="blora")
    pipeline.set_adapters(["blora"])

    # Generate
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed is not None else None
    images = pipeline(
        args.prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        generator=generator,
    ).images

    # Save
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    from slugify import slugify
    base_name = slugify(args.prompt)
    for i, img in enumerate(images):
        img.save(output_path / f'{base_name}_{i}.jpg')

    # Clean up the adapter so the pipeline can be reused with other B-LoRAs.
    pipeline.delete_adapters("blora")


if __name__ == '__main__':
    main()
