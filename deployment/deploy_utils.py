import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from dotenv import load_dotenv

def get_model_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, trust_remote_code = True, device_map = 'cuda')
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    return model, tokenizer

def check_gpu():
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f'Number of GPUs available: {num_gpus}')
        # Print the name of each GPU
        for i in range(num_gpus):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('No GPUs available, running on CPU.')

def fit_into_model(prompt, tokenizer, model):
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    outputs = model.generate(input_ids = inputs, max_new_tokens = 200)
    outputs = tokenizer.decode(outputs[0])
    return outputs

def gen_prompt(user_input, model_output, model):
    user_prompt = '<end_of_turn>\n<start_of_turn>user\n' + (user_input) + '<end_of_turn>\n<start_of_turn>model\n'
    prompt = model_output.replace('<eos>',user_prompt)
    return prompt

def get_starter_prompt(tokenizer):
    prompt = '''Play a game with me where i am in an enchanted forest full of beasts and loots and make it imaginative.
You be the narrator and i will be the player, play a dialog game with me. Let the user decide what to do next dont give options. Start The Game'''

    chat = [
        { "role": "user", "content": prompt},
        ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt
    
if __name__ == '__main__':
    print("bruh")
    # load_dotenv("/mnt/c/projects/game/token.env")
    # model_name = "google/gemma-7b-it"
    # model, tokenizer = get_model_tokenizer(model_name)
    # check_gpu()
    # prompt = get_starter_prompt()
    # print(prompt)
    # model_output = fit_into_model(prompt)
    # print(model_output)
    # user_input = "I am in a dark forest with a sword in my hand"
    # prompt = gen_prompt(user_input, model_output)
    # print(prompt)