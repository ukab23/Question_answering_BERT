import os
import pandas as pd
from deeppavlov import build_model, configs
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np

path = 'C:\\Users\\ukab2\\Downloads\\Workspace\\Sport_articles\\data\\'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

data_files = []
for f in files:
    data_files.append(f[len(path):])
# print(data_files)

titles = []
content = []
for i in data_files:
    f = open(path+i, "r")
    titles.append(i[:-4]) 
    file_text = ""
    for j in f:
        file_text += j
    content.append(file_text)
# print(titles)
# print(content)

df = pd.DataFrame()
df['title'] = titles
df['content'] = content
# print(type([df['content'][0]]))


data = []
for i in range(df.shape[0]):
    data.append(df['content'][i])

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]    

max_epochs = 10
vec_size = 100
alpha = 0.025
d2v_model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)

d2v_model.build_vocab(tagged_data)
docvec1 = d2v_model.docvecs
# print("---------------------",len(docvec1))
model = build_model(configs.squad.squad_bert, download=True)

qr = input("Enter your Question:")
test_data = word_tokenize(qr.lower())
ivec = d2v_model.infer_vector(doc_words=test_data, steps=20, alpha=0.025)


cos_score = []
for i in range(len(docvec1)):
    a = np.array(docvec1[i])
    b = np.array(ivec) 
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    cos_score.append(cos)
cos_score = pd.DataFrame(cos_score)
cos_score 
print(cos_score)
result = model([df['content'].max()], 
      [qr])

print(result)      