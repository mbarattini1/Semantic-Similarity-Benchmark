# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:25:48 2022

@author: mbara
"""

from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cos_sim(sentence1_emb, sentence2_emb):
    """
    Cosine similarity between two columns of sentence embeddings
    
    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column
    
    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
      
      source: https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e#b3f1
    """
    cosine_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cosine_sim)


def load_model(model_name):
    # Load the pre-trained model
    if 'cross-encoder' in model_name:
        return CrossEncoder(model_name)
    else:
        return SentenceTransformer(model_name)