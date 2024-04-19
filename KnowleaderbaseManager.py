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
import ntlk
from rake_nltk import  Rake
nltk.download('punkt')

class KnowleaderbaseManager:
    def __init__(self, path, embeddings=None):
        if embeddings is None:
            embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                  model_kwargs={'device': 'cuda'})
        
        self.rake_nltk_var = Rake()

        self.embeddings = embeddings
        self.file_paths = self.create_file_paths(path)

        path_knoleadgebase_path = "pathkb.json"
        file_path_path  = "filepath.json"


        try:
          self.file_paths = self.retrieve_data_from_json(file_path_path,self.embeddings)
          self.path_knowleadgebase = self.retrieve_data_from_json(path_knoleadgebase_path,self.embeddings)["allpaths"]["knowleadgebase"]

        except FileNotFoundError:
          content_data = []
          for i in self.file_paths.keys():
              file_list = []
              print(i)
              content_data.append(self.file_paths[i]['content'])
              for file_path in self.file_paths[i]['files']:
                  loader_output = self.text_loader(file_path)
                  file_list += loader_output

              self.file_paths[i]['knowleadgebase'] = self.knowleadgebase_create(file_list, self.embeddings)
          self.path_knowleadgebase = self.knowleadgebase_create(content_data, self.embeddings, 1)

          file_paths_copy = deepcopy(self.file_paths)
          self.store_data_to_json(file_paths_copy, file_path_path)
          self.store_data_to_json({"allpaths": {"knowleadgebase":self.path_knowleadgebase}}.copy(),path_knoleadgebase_path)

    # Function to serialize the knowledge base
    def serialize_knowledge_base(self,knowledge_base):
        serialized_knowledge_base = knowledge_base.serialize_to_bytes()
        encoded_knowledge_base = serialized_knowledge_base.decode('latin1')
        return encoded_knowledge_base

    # Function to deserialize the knowledge base
    def deserialize_knowledge_base(self,serialized_data,embeddings):
        knowledge_base = FAISS.deserialize_from_bytes(serialized_data.encode('latin1'),embeddings)
        return knowledge_base


    # Function to store data into JSON file
    def store_data_to_json(self,file_paths_copy, json_file):
        paths = file_paths_copy.keys()
        for path in paths:
            file_paths_copy[path]["knowleadgebase"] = self.serialize_knowledge_base(file_paths_copy[path]["knowleadgebase"])

        with open(json_file, 'w') as f:
            json.dump(file_paths_copy, f)



    # Function to retrieve data from JSON file
    def retrieve_data_from_json(self,json_file ,embeddings):
        with open(json_file, 'r') as f:
            data = json.load(f)

        paths = data.keys()
        for path in paths:
            data[path]["knowleadgebase"] = self.deserialize_knowledge_base(data[path]["knowleadgebase"],embeddings)

        return data


    def keyword_extractor(self , file_path):
        """
        Read the contents of a text file and return as a string.

        Args:
        - file_path (str): The path to the text file.

        Returns:
        - str: The content of the text file.
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()

            self.rake_nltk_var.extract_keywords_from_text(content)
            keyword_extracted = self.rake_nltk_var.get_ranked_phrases()
            content =""
            for keyword in keyword_extracted:
                content+=keyword+" "
            return content
        except FileNotFoundError:
            print("File not found.")
            return None


    def create_file_paths(self, directory):
        file_paths = {}
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)
                    if root in file_paths:
                        file_paths[root]['content'] = file_paths[root]['content'] + " " +  self.keyword_extractor(filepath)
                        file_paths[root]['files'].append(filepath)
                    else:
                        file_paths[root] = {}
                        file_paths[root]['content'] = root + "=" + self.keyword_extractor(filepath)
                        #file_paths[root]['content'] = filename
                        file_paths[root]['files'] = [filepath]

        return file_paths

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

    def knowleadgebase_API(self, q, n=5, k=5, conf=0.1):
        paths = self.path_knowleadgebase.similarity_search_with_score(q, k)

        paths_score = []
        for nor_path in paths:
            paths_score.append(nor_path[1])
            print(nor_path)

        val_paths_ind = self.p_val_gen(paths_score, conf)

        valid_paths = []
        for ind in val_paths_ind:
            valid_paths.append(paths[ind])

        if len(valid_paths) == 0:
            valid_paths = paths
        val_paths = copy.deepcopy(paths)
#      )
        det = []
        for j in range(len(paths)):
            dictionary = dict(paths[j][0])
            page_content = dictionary["page_content"]
            directory = page_content.split("=")[0]
            docs = self.file_paths[directory]['knowleadgebase'].similarity_search_with_score(q, k)
            path_deta = {}
            path_deta['path'] = page_content.split("=")[0]

            path_deta['chunkz'] = []
            sum_ = 0
            for doc_ind in range(len(docs)):
                path_deta['chunkz'].append(dict(docs[doc_ind][0])['page_content'])
#                print(docs[doc_ind][1])
                sum_ += docs[doc_ind][1]*paths[j][1]
            path_deta['score'] = sum_ / k
    
            det.append(path_deta)

        sorted_list = sorted(det, key=lambda x: x['score'])

        index_path = 0
        added_chunk = 0
        result = ""
        paths = []
        while index_path < len(sorted_list) and added_chunk <= n:
            doc_index = 0
            paths.append(sorted_list[index_path]['path'])
            result += sorted_list[index_path]['path'] + "\n"
            while doc_index < len(sorted_list[index_path]['chunkz']) and added_chunk <= n:
                result += sorted_list[index_path]['chunkz'][doc_index] + "\n"
                added_chunk += 1
                doc_index += 1
            index_path += 1
        print(paths)
        return result, paths
