from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

merged_model = AutoModelForCausalLM.from_pretrained('./merged_model')
# merged_tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

# base_model = AutoModelForCausalLM.from_pretrained('google/gemma-2b')