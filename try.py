# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/gemma-7b-it"
dtype = torch.float16
# Set your Hugging Face API token
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name,token = token)
model = AutoModelForCausalLM.from_pretrained(model_name,token = token, device_map = 'auto', torch_dtype = dtype)

input_text = 'Write a narration and provide turns for user to do action'

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = model.generate(**input_ids)

print(tokenizer.decode(output[0], skip_special_tokens=True))