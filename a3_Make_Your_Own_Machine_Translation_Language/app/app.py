from flask import Flask, render_template, request
import pickle
import torch
from models.classes import Encoder, Decoder, Seq2SeqTransformer, initialize_weights
from library.utils import get_text_transform, thtokenizer

app = Flask(__name__)

# Set device for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model metadata
with open('models/general-attention.pkl', 'rb') as f:
    meta = pickle.load(f)

# Special tokens and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Get text transformations and vocab transformations
token_transform = meta['token_transform']
vocab_transform = meta['vocab_transform']
text_transform = get_text_transform(token_transform, vocab_transform)

# Model configuration
SRC_LANGUAGE = 'input_text'  # English
TRG_LANGUAGE = 'translated_text'  # Myanmar

input_dim = len(vocab_transform[SRC_LANGUAGE])
output_dim = len(vocab_transform[TRG_LANGUAGE])

hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

# Initialize Encoder and Decoder
enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)

# Initialize the Seq2Seq model
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.apply(initialize_weights)


@app.route('/', methods=['GET'])
def index():
    """Render the initial translation page."""
    return render_template("index.html")


@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests and return results."""
    
    try:
        # Load the model weights
        print('Loading model...')
        model.load_state_dict(torch.load("models/Seq2SeqTransformer_general.pt", map_location=device))
        print('Model loaded successfully!')

        # Get the text input from the form
        prompt = request.form['query'].strip()
        print(f"Received input: {prompt}")

        if not prompt:
            return render_template('index.html', result="Please enter some text.", old_query="")

        # Transform the input text to model tokens
        src_text = text_transform[SRC_LANGUAGE](prompt).to(device)
        print(f"Tokenized input: {src_text}")  # Debug print for tokenized input
        src_text = src_text.reshape(1, -1)  # Add batch dimension

        # Create the source mask
        src_mask = model.make_src_mask(src_text)

        # Inference mode
        model.eval()
        with torch.no_grad():
            enc_output = model.encoder(src_text, src_mask)

        # Generate translation tokens
        outputs = []
        input_tokens = [SOS_IDX]  # Start token
        max_seq = 100  # Maximum sequence length

        for i in range(max_seq):
            trg_mask = model.make_trg_mask(torch.LongTensor(input_tokens).unsqueeze(0).to(device))

            output, attention = model.decoder(torch.LongTensor(input_tokens).unsqueeze(0).to(device), enc_output, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()  # Get the predicted token
            print(f"Generated token: {pred_token} ({vocab_transform[TRG_LANGUAGE].get_itos()[pred_token]})")  # Debug print for generated token

            input_tokens.append(pred_token)
            outputs.append(pred_token)

            if pred_token == EOS_IDX:  # End token, stop generating
                break

        # Convert token IDs to words
        trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]
        print(f"Generated translation tokens: {trg_tokens}")  # Debug print for the list of tokens

        # Format the translation properly
        translated_text = "".join(trg_tokens[0:-1])  # Remove <sos> and <eos> tokens

        return render_template('index.html', result=translated_text, old_query=prompt)

    except Exception as e:
        print(f"Error during translation: {e}")
        return render_template('index.html', result="An error occurred. Please try again.", old_query=prompt)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
