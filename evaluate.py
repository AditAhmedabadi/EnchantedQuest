from huggingface_hub import login
from dotenv import load_dotenv
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch
import os
from apply_format import template_from_dir

load_dotenv("token.env")
# Retrieve the API token
api_token = os.getenv("api_token")
login(api_token)

dash_line = '-' * 50

model_name = 'google/gemma-7b-it'

merged_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

test = 'Play a game with me where i am in an enchanted forest full of beasts and loots. You are the narrator. Play this game turn by turn and give me a prompt to respond'

chat = [
    {'role':'user', 'content': test}
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = merged_model.generate(input_ids=inputs.to(merged_model.device), max_new_tokens=150)
print(tokenizer.decode(outputs[0]))