from diffusers import DiffusionPipeline
import torch
from PIL import Image

print(torch.cuda.is_available())

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# prompt = "portrait of a pretty blonde woman, a flower crown, earthy makeup, flowing maxi dress with colorful patterns and fringe, a sunset or nature scene, green and gold color scheme"
# prompt = "close up photo of a rabbit, forest in spring, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot"
# prompt = "a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal"
# prompt = "freshly made jasmine tea in exquisite chinese teacup set on the table, angled shot, midday warm, Nikon D850 105mm, close-up"
prompt = "3 small Netherlands type sunflowers in a small white matt tall vase on a white table, midday warm, Nikon D850 105mm, medium shot, professional photo,"
result = pipe(prompt=prompt)
image = result.images[0]
image_path = "sunflowers.png"
image.save(image_path)

print(f"Image saved at {image_path}")