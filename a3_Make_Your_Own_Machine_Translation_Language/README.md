readme_content = """
# English-Thai Texts for Machine Translation + Transformer + Attention

This repository provides resources for building and training machine translation models between English and Thai using the **English-Thai Texts Dataset** sourced from Hugging Face.

## Dataset Overview

The **English-Thai Texts Dataset** contains parallel text data with English sentences and their corresponding translations in Thai. This dataset is ideal for training machine translation models, NLP experiments, or research related to English-Thai language processing.

- **Dataset Name**: English-Thai Texts
- **Source**: [Hugging Face Datasets - English-Thai Texts](https://huggingface.co/datasets/kvush/english_thai_texts)
- **Language Pair**: English ↔ Thai
- **Number of Samples**: 59.9k
- **Content**: The dataset contains English sentences with their corresponding Thai translations, covering various topics and contexts.

Kvush. (2024). English-Thai Texts [Dataset]. Hugging Face. https://huggingface.co/datasets/kvush/english_thai_texts

## How to Load the Dataset

To load the dataset in your Python environment, you can use the `datasets` library from Hugging Face. Here’s a quick example:

```python
from datasets import load_dataset

# Load the English-Thai dataset
dataset = load_dataset("kvush/english_thai_texts")

# Preview the data
print(dataset["train"][0])  # Print the first entry from the training set
