import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from dotenv import load_dotenv
from huggingface_hub import login 
from .deploy_utils import fit_into_model, gen_prompt, get_starter_prompt, check_gpu

# Load the environment file
load_dotenv("/mnt/c/projects/game/token.env")

# Retrieve the API token
api_token = os.getenv("api_token")

# Log in to the Hugging Face Hub
login()

# Check if CUDA is available
check_gpu()

# Define the model name
model_name = "google/gemma-7b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, trust_remote_code = True, device_map = 'cuda')
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
# tokenizer.pad_token = eos