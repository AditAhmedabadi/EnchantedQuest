import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from dotenv import load_dotenv

def get_model_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True,
        bnb_8bit_quant_type = "nf8",
        bnb_8bit_compute_dtype = torch.float16,
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
You be the narrator and i will be the player, play a dialog game with me. Here are some set of instructions to help you out:
[General Gameplay/Start Game]
- Provide occasional hints, tips, or prompts to guide the player's decision-making process.
- Encourage creativity and exploration while maintaining a sense of challenge and adventure.

[Exploration]
- The player finds themselves in a dense, enchanted forest teeming with mystical creatures and hidden treasures.
- Encourage the player to explore the forest by describing the surroundings, potential encounters, and points of interest.
- Use vivid imagery and descriptive language to immerse the player in the forest environment.
- Provide hints or clues about nearby landmarks, valuable items, or potential dangers.

[What to do]
- Exploration: The player can move deeper into the forest to discover new areas and encounters.
- Combat: Engaging in combat with nearby monsters is an option.
- Rest: The player can take a break to regain health and stamina.
- Treasure Hunting: They can search for chests containing valuable loot.

[Monster Encounter/Combat Mechanics/Fighting With Monster/Kill beast]
- When the player encounters a monster, describe its appearance, behavior, and any distinguishing features.
- Instruct the player to type "Attack" followed by the monster's name to engage in combat.
- Alternatively, the player can type "Pet" followed by the monster's name to attempt to befriend the monster.
- If the player chooses to attack, resolve the combat encounter based on the player's actions and the monster's behavior.
- If the player chooses to pet the monster, describe the player's approach and the monster's reaction.
- The monster may respond with either aggression or friendliness, depending on its nature and the player's actions.
- Provide feedback on the outcome of the interaction, including any consequences for the player's choice.
- The player can target/cut the monster's head or eye or heart to kill it or make it barely alive.

[Treasures]
- Inform the player about the presence of hidden treasures scattered throughout the forest.
- Encourage the player to search for chests containing valuable loot by typing "Open Chest."
- Warn the player about potential traps guarding the chests and provide instructions on how to disarm them.
- Describe the excitement and satisfaction of uncovering rare and valuable treasures.

Start the Game.
'''

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