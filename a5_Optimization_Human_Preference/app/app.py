from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer from Hugging Face Hub
MODEL_NAME = "KittenCat/dpo-finetuned-GPT2-RLHF-dataset"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

# Generate response function
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs, 
        max_length=200, 
        repetition_penalty=1.2,  # Penalizes repeating phrases
        temperature=0.7,  # More randomness in output
        top_k=50,  # Limits vocabulary choices
        top_p=0.9  # Nucleus sampling for diversity
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# API route for text generation
@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.json.get("prompt", "")
    if not user_input:
        return jsonify({"response": "Please enter a valid input."})
    
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
