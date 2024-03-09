import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
from apply_format import template_from_dir
from datasets import Dataset



# Load the environment file
load_dotenv("token.env")

# Retrieve the API token
api_token = os.getenv("api_token")

login(token=api_token)

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {num_gpus}')

    # Print the name of each GPU
    for i in range(num_gpus):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No GPUs available, running on CPU.')

convo_data = template_from_dir('data')

model_name = "google/gemma-2b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, token = api_token, trust_remote_code = True)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, token = api_token, trust_remote_code = True)
tokenizer.padding_side = 'right'
# tokenizer.chat_template = tokenizer.default_chat_template
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = eos

dataset = Dataset.from_dict({"chat": convo_data})
dataset = dataset.map(lambda x: {"messages": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
# print(dataset['messages'][0])
dataset = dataset.remove_columns('chat')
print(dataset)

os.environ["WANDB_MODE"]="offline"

lora_alpha = 16
lora_dropout = 0.01
lora_r = 16

peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r = lora_r,
    bias = "none",
    task_type = "CAUSAL_LM"
)

output_dir = "./training_results"
gradient_accumulation_steps = 1
save_steps = 10
logging_steps = 10
optim = "paged_adamw_32bit"
learning_rate = 2e-4
max_grad_norm = 0.3
weight_decay = 0.01
num_train_epochs = 1
lr_scheduler_type = "constant"

training_args = TrainingArguments(
    output_dir = output_dir,
    optim = optim,
    num_train_epochs = num_train_epochs,
    gradient_accumulation_steps = gradient_accumulation_steps,
    # save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    max_grad_norm = max_grad_norm,
    fp16 = True,
    group_by_length = True,
    gradient_checkpointing = False,
    weight_decay = weight_decay,
    lr_scheduler_type = lr_scheduler_type
)

max_seq_length = 500

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="messages",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

print("Definded the trainer module \nTraining.....")

trainer.train()

print("Training Completed. \nSaving the model.....")

trainer.save_model(training_args.output_dir)

print("Model saved successfully")

del model
del trainer
torch.cuda.empty_cache()

print("Model and trainer deleted from memory")

model = AutoPeftModelForCausalLM.from_pretrained(training_args.output_dir, token=api_token, trust_remote_code=True)

print("PEFT Model loaded successfully")

merged_model = model.merge_and_unload()

print("Model merged successfully")

print("Saving the merged model")
merge_output_dir = './merged_model'
merged_model.save_pretrained(merge_output_dir, safe_serialization=True)