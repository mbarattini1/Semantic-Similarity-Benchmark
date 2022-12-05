"""
preprocess text
source: https://towardsdatascience.com/the-quora-question-pair-similarity-problem-3598477af172
removed contraction expansion from source.
"""

import re
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from model_utils import SentenceSimilarityModel
import os
import numpy as np


class QuoraData():
  def __init__(self,data_loc=None,
                    preprocess=False,
                    train=None,
                    test=None):
      self.data = pd.read_csv(data_loc) if data_loc is not None else pd.concat([train,test])
      self.train = train
      self.test = test
      if preprocess:
        self.preprocess_data()
      self.similarity_model=None
      
  
  def preprocess_text(self,x):
      """
      source: https://towardsdatascience.com/the-quora-question-pair-similarity-problem-3598477af172
      removed contraction expansion from source.

      """
      #x = column.str
      x = str(x).lower()
      x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")
      x = x.replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ").replace("€", " euro ")
      x = re.sub(r"([0-9]+)000000", r"\1m", x)
      x = re.sub(r"([0-9]+)000", r"\1k", x)
      x = re.sub(r"http\S+", "", x)
      x = re.sub('\W', ' ', x)
      
      #lemmatizer = WordNetLemmatizer()
      #x = lemmatizer.lemmatize(x)
      bfs = BeautifulSoup(x)
      x = bfs.get_text()
      x = x.strip()
      return x

  def preprocess_data(self):
    print("preprocessing q1")
    self.data['question1_preprocessed'] = self.data.question1.apply(self.preprocess_text)
    print("preprocessing q2")
    self.data['question2_preprocessed'] = self.data.question2.apply(self.preprocess_text)

  def create_train_test_split(self,test_size=0.3):
    self.train, self.test = train_test_split(self.data, test_size=test_size)

  def save_split(self,save_dir):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    self.train.to_excel(save_dir+"/train.xlsx")
    self.test.to_excel(save_dir+"/test.xlsx")

  def calc_question_similarities(self,model_name=None, dataset='train',batch_size=30,**kwargs):
    """
    Calculates similarities between q1 and q2 in specified dataset.
    Args:
      model: string, encoding model name from sentence transformers library to load. If model = None, tries to use model already loded in Quora.model
      dataset: string, specifies which dataset to calculate similarities for. Either 'train', 'test', or 'data'.
    """
    self.similarity_model = SentenceSimilarityModel(model_name,**kwargs)
    #similarities = []
    #batches, num_batches = self.batch_question_pairs(dataset=dataset, batch_size = batch_size)
    #for i,(batch_q1,batch_q2) in enumerate(batches):
    #  if i % 1000 == 0: print(f"Calculating... {i} of {num_batches} batches complete")
    #  batch_similarities = self.similarity_model.get_similarity(batch_q1,batch_q2)
    #  similarities = np.append(similarities, batch_similarities)
    if dataset == 'train':
      similarities = self.similarity_model.get_similarity(self.train['question1_preprocessed'],self.train['question2_preprocessed'])
    elif dataset == 'test':
      similarities = self.similarity_model.get_similarity(self.test['question1_preprocessed'],self.test['question2_preprocessed'])
    elif dataset == 'data':
      similarities = self.similarity_model.get_similarity(self.data['question1_preprocessed'],self.data['question2_preprocessed'])
    else:
      similarities = None
      print("Invalid dataset")
      
    return similarities

  """
  def batch_question_pairs(self,dataset,batch_size):
    if dataset == 'train':
      num_batches = np.ceil(len(self.train)/batch_size)
      batches = zip(np.array_split(self.train['question1_preprocessed'].to_numpy(),num_batches),
                      np.array_split(self.train['question2_preprocessed'].to_numpy(),num_batches))
    elif dataset == 'test':
      num_batches = np.ceil(len(self.test)/batch_size)
      batches = zip(np.array_split(self.test['question1_preprocessed'].to_numpy(),num_batches),
                      np.array_split(self.test['question2_preprocessed'].to_numpy(),num_batches))
    elif dataset == 'data':
      num_batches = np.ceil(len(self.data)/batch_size)
      batches = zip(np.array_split(self.data['question1_preprocessed'].to_numpy(),num_batches),
                      np.array_split(self.data['question2_preprocessed'].to_numpy(),num_batches))
    else:
      batches = None
      print("Invalid dataset")
    return batches, num_batches
  """

  
