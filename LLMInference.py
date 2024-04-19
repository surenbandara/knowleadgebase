from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from transformers import pipeline
import torch
import json
import os
from KnowleaderbaseManager import KnowleaderbaseManager
import sys
import time

class LLAMA2Wrapper:
    def __init__(self, model_name , path):
        self.model, self.tokenizer = self.load_llama2_quantized(model_name)
        self.llama2_pipeline = self.setup_pipeline()
        self.knowleadgebasemanager = KnowleaderbaseManager(path)

    def load_llama2_quantized(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",       
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def setup_pipeline(self):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1000,
            temperature=0.75,
            top_k=10,         
            top_p=0.9,
            batch_size=1,
            do_sample=True
        )

    def generate_response_prompt(self, query, context):
        text = """INSTRUCTION: Generate the Output for the given Query considering the given Context. First search and >        to generate the most accurate answer for the Query. Then combine the collected data in a meaningful way to get >
        Do not consider the data that is not present in the Context for Output generation.

        If you cannot find required data in the Context to generate the Output, give 'No data match' as the Output.

        The given Context is a combination of multiple text chunks from different sources. These text chunks are separa>
        You are required to produce the most accurate response based on the given Context. Don't add unwanted details i>        """
        text += f'QUERY: \n{query}\n\n'
        text += f'CONTEXT: \n{context}\n\n'
        text += 'OUTPUT: '
        return {'text': text}

    def llama2_response(self):
        while True:
            query = input("Enter your query (type 'quit' to exit): ")
            if query.lower() == 'quit':
                print("Exiting...")
                break

            # Display loading animation
            loading_chars = ['|', '/', '-', '\\']
            loading_text = "Generating response... "
            for char in loading_chars:
                sys.stdout.write(loading_text + char + '\r')
                sys.stdout.flush()
                time.sleep(0.1)
            
            #keywords = self.llama2_pipeline('QUERY :'+query+ "\nCONTEXT: The company is about sri lanka central bank , i need keywords for this query"
            #"\nOUTPUT: ")[0]['generated_text'].split('\nOUTPUT:')[1]
            
            #print(keywords)
            context = self.knowleadgebasemanager.knowleadgebase_API(query,3,5)[0]
            output_response = self.llama2_pipeline(
                self.generate_response_prompt(query, context)['text'])[0]['generated_text']
            output_text = output_response.split('\nOUTPUT:')[1]

            # Clear loading animation
            sys.stdout.write(' ' * len(loading_text) + '\r')
            sys.stdout.flush()

            print("Generated response:")
            print(output_text)


