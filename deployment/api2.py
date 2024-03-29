from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
# Initialize Flask app
app = Flask(__name__)

model_name = "google/gemma-7b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config ,trust_remote_code = True, device_map = 'cuda')
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

# Define function to generate text using the model directly
def generate_text(prompt, model, tokenizer, max_length=100, temperature=0.5):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    # Generate text based on the prompt
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        # num_return_sequences=1,  # Adjust as needed
        # pad_token_id=tokenizer.eos_token_id,  # End of sequence token
        early_stopping=False  # Stop generation when EOS token is generated
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# API endpoint for generating text
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')

    print(f"Received prompt: {prompt}")
    
    # Generate text using the provided prompt
    generated_text = generate_text(prompt, model, tokenizer)
    
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)