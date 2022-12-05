# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:25:48 2022

@author: mbara
"""

from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from numpy.linalg import norm

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

class SentenceSimilarityModel():
    def __init__(self,model_name,**kwargs):
        self.model = self.load_model(model_name,**kwargs)
        self.model_name = model_name
        self.model_type = 'cross-encoder' if 'cross-encoder' in model_name else 'bi-encoder'

    def load_model(self,model_name,**kwargs):
        # Load the pre-trained model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if 'cross-encoder' in model_name:
            return CrossEncoder(model_name,**kwargs,device=device)
        else:
            return SentenceTransformer(model_name,**kwargs,device=device)


    def create_embeddings(self,q1,q2):
      sentence1_emb = self.model.encode(q1, show_progress_bar=True)
      sentence2_emb = self.model.encode(q2, show_progress_bar=True)
      return sentence1_emb, sentence2_emb

    def get_similarity(self,s1,s2):
        if self.model_type == 'cross-encoder':
            paired_sentences = self.pair_inputs(s1.astype(str).to_numpy(),s2.astype(str).to_numpy())
            similarities = self.model.predict(paired_sentences)
        else:
          s1_embed, s2_embed = self.create_embeddings(s1,s2)
          similarities = self.cosine_similarities(s1_embed,s2_embed)
        return similarities
        
    def pair_inputs(self,s1,s2):
        sentence_pairs=[]
        for sentence1, sentence2 in zip(s1,s2): sentence_pairs.append([sentence1, sentence2])
        return sentence_pairs
    
    def cosine_similarities(self,A,B):
      #also works
      sims = [np.dot(A[i,:],np.transpose(B[i,:]))/(norm(A[i,:])*norm(B[i,:])) for i in range(A.shape[0])]
      return sims
      
      


