from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("./data/ddpm-butterflies-128")
image = pipe(num_inference_steps=50).images[0]
image.show()