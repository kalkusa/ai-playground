import logging
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os, torch

# Suppressing warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuring logging
logging.basicConfig(level=logging.ERROR)



# Setting environment variable to improve memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = 'cuda'
model_name = "speakleash/Bielik-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the model in bfloat16 precision to reduce memory usage and move it to the GPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0) 

prompt = "Kto ty jesteś?"

sequences = text_generator(prompt, max_new_tokens=50, do_sample=True, eos_token_id=tokenizer.eos_token_id, top_k=50)

print(f'User: {prompt}')
for seq in sequences:
    print(f"Bielik: {seq['generated_text']}")
