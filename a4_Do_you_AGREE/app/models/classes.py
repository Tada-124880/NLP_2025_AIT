import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

# Define the NLI Model class
class NLIModel:
    def __init__(self):
        # Load the tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.load_state_dict(torch.load('models/bert_model.pth', map_location=torch.device('cpu')), strict=False)
        self.model.eval()

    # Helper function for mean pooling
    def mean_pool(self, token_embeds, attention_mask):
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool

    # Helper function for cosine similarity
    def cosine_similarity_func(self, u, v):
        return cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    # NLI prediction function
    def nli_prediction(self, premise, hypothesis):
        # Tokenize input sentences
        inputs_a = self.tokenizer(premise, return_tensors='pt', truncation=True, padding=True)
        inputs_b = self.tokenizer(hypothesis, return_tensors='pt', truncation=True, padding=True)

        # Extract token embeddings from BERT
        u = self.model(inputs_a['input_ids'])[0]
        v = self.model(inputs_b['input_ids'])[0]

        # Get the mean-pooled vectors
        u_mean = self.mean_pool(u, inputs_a['attention_mask']).detach().cpu().numpy().reshape(-1)
        v_mean = self.mean_pool(v, inputs_b['attention_mask']).detach().cpu().numpy().reshape(-1)

        # Calculate cosine similarity
        similarity = self.cosine_similarity_func(u_mean, v_mean)

        # Return NLI prediction based on similarity score
        if similarity > 0.7:
            return "Entailment"
        elif similarity < 0.5:
            return "Contradiction"
        else:
            return "Neutral"
