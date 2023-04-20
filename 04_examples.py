import torch
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline

device = "cuda"
input_dir = "input/"
output_dir = "output/"


def unconditional_image_generation():
    generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128")
    generator.to(device)
    image = generator().images[0]
    image.save(output_dir + "unconditional.png")


def conditional_image_generation():
    generator = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
    generator.to(device)
    image = generator("An image of a squirrel in Picasso style").images[0]
    image.save(output_dir + "conditional.png")


def text_guided_img_to_img():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16).to("cuda")
    prompt = "ghibli style, a fantasy landscape with castles"
    generator = torch.Generator(device=device).manual_seed(1024)
    init_image = Image.open(input_dir + "sketch-mountains-input.jpg")
    image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
    image.save(output_dir + "guided_img_to_img.png")


def text_guided_depth_to_image():
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        torch_dtype=torch.float16,
    ).to(device)
    init_image = Image.open(input_dir + "forrest_input.jpg")
    prompt = "river in the forrest"
    n_prompt = "trees in water"
    image = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=0.7).images[0]
    image.save(output_dir + "guided_depth_img_2.png")


if __name__ == "__main__":
    text_guided_depth_to_image()