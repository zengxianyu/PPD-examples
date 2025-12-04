from structured_noise import generate_structured_noise_batch_vectorized
import argparse
import torch
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import download_models

def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=False,
        default="models/ppd/flux1-dev_lora_color_step=266000_biased.safetensors",
        help="Path to lora checkpoint file"
    )
    parser.add_argument(
        "--radius",
        type=int,
        required=False,
        default=35,
        help="Radius of structured noise"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="ppd/test1.jpg",
        help="Input image filename"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output.png",
        help="Output image filename"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A high quality scene from a movie captured by a professional camera. A woman stands in a rugged cave-like environment with stone walls and patches of snow. She has an adventurous appearance, wearing a sleeveless top, shorts, gloves, and a utility belt, exuding a determined and alert expression as if ready to explore or face a challenge"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="ugly, low quality, CG, Render, unreal, game, cartoon, blur, low res",
        help="Negative prompt"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_models(["FLUX.1-dev"])
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    embed_layers = None

    pipe.load_lora(pipe.dit, args.lora_checkpoint_path, alpha=1)

    image_in_pil = Image.open(args.input_image).convert("RGB")
    w,h = image_in_pil.size
    if args.height is not None and args.width is not None:
        use_original_size = False
        new_w, new_h = args.width, args.height
    else:
        use_original_size = True
        new_w, new_h = w//16*16, h//16*16
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)
    prompt = args.prompt
    with torch.no_grad():
        image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=False)

        radius = args.radius
        input_noise = torch.randn_like(input_latents)
        noise = generate_structured_noise_batch_vectorized(input_latents, cutoff_radius=radius, input_noise=input_noise)
        noise = noise.contiguous()

        negative_prompt = args.negative_prompt

        image = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            height=new_h, width=new_w,
            cfg_scale=2, num_inference_steps=50, noise=noise
        )

        if use_original_size:
            image = image.resize((w,h))
        image.save(args.output_name)

