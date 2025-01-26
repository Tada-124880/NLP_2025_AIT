import numpy as np

class Gensim():
    def __init__(self, model):
        self.model = model

    def get_embed(self, word):

        default_vector = np.zeros(self.model.vector_size)
        
        try: 
            result = self.model.get_vector(word.lower().strip())
        except:
            result = default_vector

        return result