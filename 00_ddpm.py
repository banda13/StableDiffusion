from diffusers import DDPMPipeline


ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
image = ddpm(num_inference_steps=50).images[0]
image.show()