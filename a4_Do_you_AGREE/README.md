

# Task 1. Training BERT from Scratch

## Overview
This project implements **BERT (Bidirectional Encoder Representations from Transformers)** from scratch, following the **Masked Language Model (MLM)** and **Next Sentence Prediction (NSP)** objectives. The model is trained using a subset of the **BookCorpus dataset** and tested for inference.

## 1. Dataset Details
- **Dataset Used:** BookCorpus (subset of 80,000 sentences)
- **Source** BookCorpus. (n.d.). Retrieved from [Hugging Face Datasets](https://huggingface.co/datasets/bookcorpus). This dataset contains a large collection of books.

- **Preprocessing:**
  - Tokenization of sentences using a custom tokenizer
  - Conversion of words into numerical token IDs
  - Padding to ensure uniform sequence lengths
  - Splitting data into sentence pairs for the NSP task

## 2. Model Architecture
The implemented BERT model follows the original structure with the following key components:

- **Embeddings:**
  - **Token Embeddings:** Convert words into dense vector representations
  - **Segment Embeddings:** Distinguish between different input sentences in the NSP task
  - **Positional Embeddings:** Encode the order of words within a sequence

- **Transformer Encoder:**
  - **Multi-head Self-Attention Mechanism**
  - **Scaled Dot-Product Attention**
  - **Feed-forward Networks**
  - **Layer Normalization and Dropout**

## 3. Training Process
### **Hyperparameters:**
- **Batch Size:** 6
- **Sequence Length:** 1000
- **Embedding Size:** 768
- **Hidden Layers:** 12
- **Attention Heads:** 12
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Number of Epochs:** 1000

### **Training Loss:**
- Final Loss : **3.956977**

### **Saving Model Weights:**
```python
import torch

# Save model
torch.save(model.state_dict(), "bert_model.pth")

# Load model
model.load_state_dict(torch.load("bert_model.pth"))
```

## 4. Inference
The model is evaluated using masked token predictions and next sentence prediction. Below is an example inference result:

### **Input Sentence:**
```plaintext
['[CLS]', 'it', "'s", 'not', 'like', 'he', "'s", 'using', 'it', 'much', 'these', 'days', '[SEP]', '``', 'i', 'never', 'said', 'i', 'did', "n't", 'want', 'you', 'two', 'to', 'get', 'to', 'know', '[MASK]', 'otheri', 'said', 'i', 'did', '[MASK]', 'want', '[MASK]', '[MASK]', 'him', 'for', 'a', 'fling', "''", '[SEP]']
```

### **Masked Tokens (Original):**
```plaintext
["n't", 'you', 'each', 'days', 'using']
```

### **Masked Tokens (IDs):**
```plaintext
[14707, 18180, 18304, 4037, 5460]
```

### **Predicted Masked Tokens:**
```plaintext
['[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
```

### **Next Sentence Prediction:**
- **Actual:** False
- **Predicted:** True

### **Interpretation of NSP Result**
- **Ground Truth (isNext: False):** The second sentence was **not** the actual next sentence in the dataset.
- **Model Prediction (predict isNext: True):** The model incorrectly predicted it as the correct next sentence.

Possible reasons for the error:
1. **Undertrained Model** – At **epoch 1000**, loss is still relatively high (**3.956977**), indicating the model may not have learned sentence relationships well.
2. **Token Masking Impact** – Missing words may confuse the model’s understanding of sentence coherence.
3. **Limited Dataset** – Training on **80,000 sentences** is small compared to the original BERT dataset, affecting generalization.

### **Improving NSP Accuracy**
- **More Training:** Increasing the number of epochs.
- **Larger Dataset:** Using additional text data to improve learning.
- **Hyperparameter Tuning:** Adjusting learning rates and batch sizes.

### **Limitation**
- **Computational Resources:** The lack of access to high-performance computing resources, such as GPUs or specialized hardware, restricts the ability to efficiently train large-scale models. This limitation impacts the training time and overall performance, especially for resource-intensive tasks like BERT model training.
---

