from flask import Flask, render_template, request
import torch
import pickle
from torchtext.data.utils import get_tokenizer
from models.def_model import LSTMLanguageModel
from library.gen import generate

# Initialize Flask app
app = Flask(__name__)

# Load vocabulary and model
with open('../app/models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
emb_dim, hid_dim, num_layers, dropout_rate = 1024, 1024, 2, 0.65  # Model hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate and load the LSTM language model
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('models/best-val-lstm_lm.pt', map_location=device))

# Tokenizer
tokenizer = get_tokenizer('basic_english')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate_hogwart():
    # Get user input prompt from the form
    prompt = request.form['query'].strip()

    # Model parameters for text generation
    max_seq_len = 30
    temperature = 0.8  # Control creativity of the output (lower = more predictable)

    # Generate text based on the prompt
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device)

    # Join the list of generated words into a single string
    result = " ".join(generation)

    return render_template('index.html', result=result, old_query=prompt)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
