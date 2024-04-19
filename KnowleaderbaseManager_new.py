from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters.base import TextSplitter
import pandas as pd
import os
from langchain_community.document_loaders.text import TextLoader

import numpy as np
from scipy.stats import norm
from copy import deepcopy

import json
import base64

import copy

class KnowleaderbaseManager:
    def __init__(self, path, embeddings=None):
        if embeddings is None:
            embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                  model_kwargs={'device': 'cuda'})
        self.embeddings = embeddings


    
        index_path = "knowleadgebase_index"
        try:
            self.knowleadgebase = self.load_faiss_index(index_path, self.embeddings)
        except :
            self.knowleadgebase = self.create_file_paths(path)
            self.save_faiss_index(index_path)
          



    def create_file_paths(self, directory):
        directory_paths =[]
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)
                    print(filepath)
                    directory_paths+=self.text_loader(filepath)
        
        knowleadgebase = self.knowleadgebase_create(directory_paths,self.embeddings)
        return knowleadgebase
    
    def save_faiss_index(self, index_path):
        self.knowleadgebase.save_local(index_path)

    def load_faiss_index(self, index_path, embeddings):
        return FAISS.load_local(index_path, embeddings)

    def text_loader(self, path, encoding='utf-8'):
        loader = TextLoader(path, autodetect_encoding=True)
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=900,
            length_function=len,
            is_separator_regex=False
        )
        return loader.load_and_split()

    def knowleadgebase_create(self, text_list, embeddings, status=0):
        if status == 0:
            knowledgeBase = FAISS.from_documents(text_list, embeddings)
        else:
            knowledgeBase = FAISS.from_texts(text_list, embeddings)
        return knowledgeBase

    @staticmethod
    def p_val_gen(data, conf=0.1):
        mean = np.mean(data)
        std_dev = np.std(data)
        normal_dist = norm(loc=mean, scale=std_dev)
        p_values = [normal_dist.cdf(x) for x in data]
        valid_set = []
        for p in range(len(p_values)):
            if p_values[p] <= conf:
                valid_set.append(p)
        return valid_set

    def knowleadgebase_API(self, q,k=3, n=5, conf=0.1):
        paths = self.knowleadgebase.similarity_search_with_score(q, n)
        
        result =""
        for i in paths:
            result+= dict(i)["page_content"]
        return result, result
