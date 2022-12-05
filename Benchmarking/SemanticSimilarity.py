# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:29:03 2022

@author: mbara
"""
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import CrossEncoder


class Benchmark():
    def __init__(self,task_name,language='en'):
        self.task = task_name.upper()
        self.language = language
        
        if self.task == 'STS':
            self.dataset = load_dataset('stsb_multi_mt', self.language)
            self.train = pd.DataFrame(self.dataset['train'])
            self.test = pd.DataFrame(self.dataset['test'])
            self.input_columns = [col for col in self.test.columns.tolist() if 'sentence' in col]
            self.label_column ='similarity_score'
        
        elif self.task == 'SICK':
            self.dataset = load_dataset('sick')
            self.train = pd.DataFrame(self.dataset['train'])
            self.test = pd.DataFrame(self.dataset['test'])
            self.input_columns = ['sentence_A','sentence_B']
            self.label_column ='relatedness_score'
        else:
            print("INVALID TASK")
            
        self.output_columns = set()
        self.results = pd.DataFrame(columns=['model','pearson'])
            


    def benchmark_models(self,model_names):
        for model in model_names:
            if 'cross-encoder' in model:
                self.predict_similarity(model)
            else:
                self.compute_cosine_similarity(model)
        self.eval_models()
        
    def add_output_columns(self,names):
        for name in names: self.output_columns.add(name)
        
    def compute_cosine_similarity(self,model_name):
        model = self.load_model(model_name)
        sentence1_emb = model.encode(self.test[self.input_columns[0]], show_progress_bar=True)
        sentence2_emb = model.encode(self.test[self.input_columns[1]], show_progress_bar=True)
        self.test[model_name] = np.diag(cosine_similarity(sentence1_emb,sentence2_emb))
        self.add_output_columns([model_name])
        
    def predict_similarity(self,model_name):
        model = self.load_model(model_name)
        sentence_pairs=[]
        for sentence1, sentence2 in zip(self.test[self.input_columns[0]], self.test[self.input_columns[1]]):
            sentence_pairs.append([sentence1, sentence2])
        self.test[model_name] = model.predict(sentence_pairs, show_progress_bar=True)
        self.add_output_columns([model_name])
        
        
    def eval_models(self):
        score_cols = [self.label_column] + list(self.output_columns)
        res = self.test[score_cols].corr(method='pearson').iloc[1:, 0:1].transpose()
        res = res.rename(columns={self.label_column:'Pearsons R'})
        self.results = res
        return self.results

    def plot_model_correlations(self):
        for model_column in self.output_columns:
            plt.figure()
            plt.scatter(self.test[self.label_column],self.test[model_column])
            plt.ylabel('Model Similarity')
            plt.xlabel('Similarity Label')
            plt.title(f"{model_column}\n Pearson: {self.results[model_column].item()}\n")
            filename = model_column.replace("/","-")
            plt.savefig(f"results/{self.task}/{filename}_results.png")
            plt.close()
            
    def load_model(self,model_name):
        # Load the pre-trained model
        if 'cross-encoder' in model_name:
            return CrossEncoder(model_name)
        else:
            return SentenceTransformer(model_name)



# sentence_pairs = []
# for sentence1, sentence2 in zip(stsb_test['sentence1'], stsb_test['sentence2']):
#     sentence_pairs.append([sentence1, sentence2])
    
# stsb_test['SBERT CrossEncoder_score'] = model.predict(sentence_pairs, show_progress_bar=True)
    
        
if __name__ == "__main__":
    benchmark_models = ['cross-encoder/stsb-distilroberta-base','stsb-mpnet-base-v2','all-mpnet-base-v2','stsb-distilroberta-base-v2','all-distilroberta-v1',]

    sts  = Benchmark('STS')
    sts.benchmark_models(benchmark_models)
    sts.results.to_excel("results/STS/STS_benchmark_results.xlsx")
    sts.test.to_excel("results/STS/STS_benchmark_outputs.xlsx")
    sts.plot_model_correlations()
        
    sick = Benchmark('SICK')
    sick.benchmark_models(benchmark_models)
    sick.results.to_excel("results/SICK/SICK_benchmark_results.xlsx")
    sick.test.to_excel("results/SICK/SICK_benchmark_outputs.xlsx")
    sick.plot_model_correlations()
        