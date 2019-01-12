#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nozomi
"""
from gensim.models import word2vec
import csv
import numpy as np
import collections
import nltk
import tltk
from pythainlp.tokenize import word_tokenize

def tokenizer_tltk(text): #(ไม่ใช้แล้ว)
    """
    use tltk.nlp.word_segment
    output is like this:
    'กรุงเทพมหานคร|<s/>เป็น|เมืองหลวง|และ|นคร|ที่|มี|ประชากร|มาก|ที่สุด|ของ|ประเทศ|ไทย|<s/>'
    split with '|' and cut <s/>, <u/>, Fail > return list of words
    """
    tokens = tltk.nlp.word_segment(text).split('|')
    word_list = []
    for token in tokens:
        reshaped = token.strip('<s/>')
        reshaped = reshaped.strip('<u/>')
        reshaped = reshaped.strip(' ')
        reshaped = reshaped.strip('Fail>')
        reshaped = reshaped.strip('</Fail')
        if reshaped != '':
            word_list.append(reshaped)
    return word_list

def tokenizer(text):
    word_list = word_tokenize(text)
    return word_list

def tokenize_headline(start_index, end_index, open_tsv='thairath.tsv', write_tsv='headline.tsv'):
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
    
    new_list = []
    for line in lines[start_index: end_index + 1]:
        if line[0] not in id_list:
            new_line = [line[0], '|'.join(tokenizer(line[1]))]
            new_list.append(new_line)
            
    writer = csv.writer(write_file, lineterminator='\n', delimiter='\t')
    writer.writerows(new_list)
    
    open_file.close()
    write_file.close()
    
def tokenize_article(start_index, end_index, open_tsv='thairath.tsv', write_tsv='article.tsv'):
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
    
    new_list = []
    for line in lines[start_index: end_index + 1]:
        if line[0] not in id_list:
            new_line = [line[0], '|'.join(tokenizer(line[-1]))]
            new_list.append(new_line)
            
    writer = csv.writer(write_file, lineterminator='\n', delimiter='\t')
    writer.writerows(new_list)
    
    open_file.close()
    write_file.close()

def make_model(open_tsv='article.tsv', save_name='article.model'):
    file = open(open_tsv, 'r', encoding='utf-8')
    lines = list(csv.reader(file, delimiter='\t'))
    word_list = [line[1].split('|') for line in lines]
    model = word2vec.Word2Vec(word_list, size=200, min_count=5, window=15)
    model.save(save_name)
    
def similar(word, model='article.model'):
    model = word2vec.Word2Vec.load(model)
    results = model.wv.most_similar(positive=[word])
    for result in results:
       print(result)
       
def calc(posi, nega, model='article.model'):
    model = word2vec.Word2Vec.load(model)
    results = model.wv.most_similar(positive=posi, negative=nega)
    for result in results:
       print(result)