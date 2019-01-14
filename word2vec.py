#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nozomi
"""
from gensim.models import word2vec
import csv
import numpy as np
from pythainlp.tokenize import word_tokenize
def tokenizer(text):
    word_list = word_tokenize(text)
    return word_list
    
def tokenize(start_index, end_index, open_tsv='thairath.tsv', write_tsv='article.tsv'):
    """
    tokenize only headline and save
    """
    # make id list for checking duplicate
    file = open(write_tsv, 'r', encoding='utf-8')
    lines = list(csv.reader(file, delimiter='\t'))
    id_list = [line[0] for line in lines]
    file.close()
    
    open_file = open(open_tsv, 'r', encoding='utf-8')
    write_file = open(write_tsv, 'a', encoding='utf-8')  # append mode
    lines = list(csv.reader(open_file, delimiter='\t'))
    writer = csv.writer(write_file, lineterminator='\n', delimiter='\t')
    
    for line in lines[start_index: end_index + 1]:
        if line[0] not in id_list:
            # ถ้าใช้ headline line[1], description line[2] 
            new_line = [line[0], '|'.join(tokenizer(line[-1]))]
            writer.writerow(new_line)
    
    open_file.close()
    write_file.close()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def make_model(open_tsv='article.tsv', save_name='article.model'):
    file = open(open_tsv, 'r', encoding='utf-8')
    lines = list(csv.reader(file, delimiter='\t'))
    word_list = [line[1].split('|') for line in lines]
    model = word2vec.Word2Vec(word_list, size=200, min_count=5, window=15)
    model.save(save_name)
    
def similar(word, n=20, model='article.model'):
    model = word2vec.Word2Vec.load(model)
    results = model.wv.most_similar(positive=[word], topn=n)
    for result in results:
       print(result)
       
def calc(posi, nega, n=20, model='article.model'):
    model = word2vec.Word2Vec.load(model)
    results = model.wv.most_similar(positive=posi, negative=nega, topn=n)
    for result in results:
       print(result)
       
def metonymy_vec(open_tsv='wordpair.tsv',write_tsv='metonymy_vector.tsv', model='article.model'):
    model = word2vec.Word2Vec.load(model)
    open_file = open(open_tsv, 'r', encoding='utf-8')
    write_file = open(write_tsv, 'w', encoding='utf-8')
    lines = list(csv.reader(open_file, delimiter='\t'))
    writer = csv.writer(write_file, lineterminator='\n', delimiter='\t')
    
    for line in lines:
        new_line = line + list(model.wv[line[0]]-model.wv[line[1]])
        writer.writerow(new_line)
        
    open_file.close()
    write_file.close()   
    
def metonymy_sim(metonymy1, country1, metonymy2, country2, model='article.model'):
    model = word2vec.Word2Vec.load(model)
    vec1 = model.wv[metonymy1]-model.wv[country1]
    vec2 = model.wv[metonymy2]-model.wv[country2]
    return cos_sim(vec1, vec2)