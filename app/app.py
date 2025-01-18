from flask import Flask, render_template, request
import numpy as np
import pickle
import nltk
from nltk.corpus import brown
from model.def_gensim import Gensim

# Initialize Flask app
app = Flask(__name__)

# Load model and corpus
model_gensim = Gensim(pickle.load(open('model/GloVe_gensim.model', 'rb')))
nltk.download('brown')
corpus = brown.sents(categories="news")

# Utility function to compute cosine similarities and get closest sentences
def find_closest_sentences(corpus_embeds, query_embed, k=10):
    similarities = np.dot(corpus_embeds, query_embed) / (np.linalg.norm(corpus_embeds, axis=1) * np.linalg.norm(query_embed))
    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", models={'gensim': 'GloVe (Gensim)'})

@app.route('/search', methods=['POST'])
def search():
    # Get query and model choice from form
    query = request.form['query'].strip()

    # Embed query and corpus sentences
    query_embed = np.mean([model_gensim.get_embed(word) for word in query.split()], axis=0)
    corpus_embeds = np.array([np.mean([model_gensim.get_embed(word) for word in sentence], axis=0) for sentence in corpus])

    # Find closest sentences
    closest_idx = find_closest_sentences(corpus_embeds, query_embed)

    # Prepare result
    result = [' '.join(corpus[idx]) for idx in closest_idx]

    return render_template('index.html', models={'gensim': 'GloVe (Gensim)'}, result=result, model_name='GloVe (Gensim)', old_query=query)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
