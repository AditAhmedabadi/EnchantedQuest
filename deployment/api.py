from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
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

generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        return_tensors='pt',
        max_new_tokens=200,
        model_kwargs={"torch_dtype": torch.bfloat16}
)



# @app.route("/generate", methods=["POST"])
# def generate_text():
#     data = request.json
#     prompt = data["prompt"]
#     # previous_response = data.get("previous_response", "")
#     max_length = data.get("max_length", 50)
#     # prompt += " " + previous_response  # Append previous response to the prompt
#     response = generator(prompt, max_length=max_length)
#     token_ids = response[0]["generated_token_ids"]
#     response_text = tokenizer.decode(token_ids)
#     return jsonify(response_text)

if __name__ == "__main__":
    # app.run(debug=True)
    generated_tokens = generator("Describe Life in the 21st century.")[0]['generated_token_ids']
    print(tokenizer.decode(generated_tokens))