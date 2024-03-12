from diffusers import DiffusionPipeline
import torch
from PIL import Image

print(torch.cuda.is_available())

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

prompt = "A monkey in front of computer with software development editor on the screen. 16k, ultra photo realism, golden hour."
result = pipe(prompt=prompt)
image = result.images[0]
image_path = "monke.png"
image.save(image_path)

print(f"Image saved at {image_path}")