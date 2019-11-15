import os
import pandas as pd
from deeppavlov import build_model, configs
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np

class Classifier():
    def __init__(self):
        self.model = build_model(configs.squad.squad_bert, load_trained=True)        

    def get_answer(self, para, qr):
        ans = self.model([para],[qr]) 
        return ans

if __name__ == '__main__':
    result = Classifier()
    para = input("Paragraph:")
    while(True):
        qr = input("Question:")
        if qr == "exit":
            break
        else:
            print(result.get_answer(para,qr))