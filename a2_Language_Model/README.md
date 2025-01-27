# Harry Potter Language Model

This repository contains a language model built using an LSTM architecture that learns sequential patterns from a dataset of plotlines and character dialogues from the Harry Potter movies. The model is trained to predict the next word in a sequence, enabling it to generate coherent text and understand the structure of conversations from the Harry Potter universe.

## Dataset
The dataset used in this project consists of plots and dialogues between characters from the Harry Potter movies.

- Source: Harry Potter Tiny Dataset on Hugging Face
- Description: The dataset contains text-based interactions between characters, including plotlines and dialogues from the Harry Potter movies.

Mickume. (2022). *Harry Potter Tiny* [Dataset]. Hugging Face. https://huggingface.co/datasets/mickume/harry_potter_tiny

### How to Acquire the Dataset
To acquire the dataset, simply download it using Hugging Face’s datasets library.

from datasets import load_dataset

dataset = load_dataset("mickume/harry_potter_tiny")

## Model Training
The model is trained on the tokenized text from the Harry Potter dataset using an LSTM-based architecture. Below are the detailed steps and components involved in training the model.

### 1. Tokenization
#### 1.1) Tokenizer Definition
We use the basic_english tokenizer from torchtext, which performs the following steps:

- Lowercasing all words
- Removing punctuation
- Tokenizing based on spaces and other delimiters

import torchtext

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

#### 1.2) Tokenization Function
A function is defined to tokenize each example in the dataset and return it in dictionary format.

tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}

#### 1.3) Apply Tokenization
The tokenization function is applied to the dataset, and the 'text' column is removed in the process.

tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})

#### 1.4) Building the Vocabulary
A vocabulary is built from the tokenized data. We use a minimum frequency of 3 to ensure that only words that appear frequently enough are included.

vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'], min_freq=3)

### 2. Model Architecture
#### 2.1) LSTM-based Language Model
The model architecture consists of the following layers:

1. Embedding Layer: Each word is mapped to a vector representation.
2. LSTM Layer: A stacked LSTM with 2 layers, each containing 1024 hidden units, designed to capture long-range dependencies in sequential data. Dropout is applied to prevent overfitting.
3. Loss Function: Cross-entropy loss is used for predicting the next token in the sequence.
4. Model Parameters: The model contains trainable parameters for the embedding layer, LSTM layers, and output layer.

### 2.2) Training Process
The training process involves:

- Data Batching: Feeding batches of tokenized data through the model.
- Loss Calculation: The model computes the cross-entropy loss based on the predicted and actual next words.
- Backpropagation: The model’s parameters are updated using backpropagation.
- Learning Rate Adjustment: The learning rate is adjusted based on validation performance.
- Model Evaluation: The model is evaluated using perplexity, a metric that indicates how well the model predicts unseen data.

## Requirements
- Python 3.8+
- PyTorch
- Torchtext
- Hugging Face datasets library

## Usage
1. Clone the repository.
2. Install dependencies using pip install -r requirements.txt.
3. Acquire the dataset using the Hugging Face datasets library.
4. Run the training script to train the language model on the dataset.

## Results
After training, the model can be used to generate text or predict the next word in a sequence based on the Harry Potter dialogues and plotlines. The performance is evaluated using perplexity, with lower perplexity indicating better model performance.