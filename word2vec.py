# -*- coding: utf-8 -*-
"""
@author: Nozomi
"""
from gensim.models import word2vec
import csv
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
font = {"family":"Ayuthaya"}
mpl.rc('font', **font)
import matplotlib.pyplot as plt
from pythainlp.tokenize import word_tokenize

def tokenizer(text):
    word_list = word_tokenize(text)
    return word_list
    
def tokenize(start_index, end_index, open_tsv='thairath1.tsv', write_tsv='tokenized1.tsv'):
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
    
    for line in lines[start_index: end_index]:
        if line[0] not in id_list:
            # ถ้าใช้ headline line[1], description line[2] 
            new_line = [line[0], '|'.join(tokenizer(line[-1]))]
            writer.writerow(new_line)
    
    open_file.close()
    write_file.close()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mahalanobis(vectors):
    mean_vec = np.mean(vectors, axis=0)
    deviation_vec = vectors - mean_vec
    cov_matrix = np.cov(vectors.T, bias=False)
    inv_matrix = np.linalg.inv(cov_matrix)
    mahal_dis = list(map(lambda vec: np.sqrt(np.dot(np.dot(vec, inv_matrix), vec.T)), deviation_vec))
    return mahal_dis

def make_model(open_tsv='tokenized1.tsv', save_name='article.model'):
    file = open(open_tsv, 'r', encoding='utf-8')
    lines = list(csv.reader(file, delimiter='\t'))
    word_list = [line[1].split('|') for line in lines]
    model = word2vec.Word2Vec(word_list, size=200, min_count=5, window=15)
    model.save(save_name)

    
class Metonymy:

    def __init__(self, model='article.model'):
        self.model = word2vec.Word2Vec.load(model)
        self.vocab = list(self.model.wv.vocab.keys())
    
    def similar(self, word, n=20):
        results = self.model.wv.most_similar(positive=[word], topn=n)
        for result in results:
           print(result)
           
    def sim_two_word(self, word1, word2):
        return cos_sim(met.model.wv[word1], met.model.wv[word2])
    
    def dis_two_word(self, word1, word2):
        return np.linalg.norm(met.model.wv[word1] - met.model.wv[word2])
    
    def dis_two_word_random(self):
        words = random.sample(self.vocab, 2)
        #print(words)
        return (np.linalg.norm(met.model.wv[words[0]] - met.model.wv[words[1]]))
    
    def dis_words(self, num_of_pair, log=False):
        dis_list = [self.dis_two_word_random() for i in range(num_of_pair)]
        
        plt.hist(dis_list, bins=240, range=(0,60))
        plt.xlabel('Euclidean distance')
        plt.ylabel('Numbers')
        if log == True:
            plt.yscale('log')
        plt.title('distances of random {} word pairs (bin=0.25)'.format(num_of_pair))
        plt.show()
        
    def calc(self, posi, nega, n=20):
        results = self.model.wv.most_similar(positive=posi, negative=nega, topn=n)
        for result in results:
           print(result)
           
    def save_vec(self, open_tsv='wordpair.tsv',write_tsv='metonymy_vector.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        write_file = open(write_tsv, 'w', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        writer = csv.writer(write_file, lineterminator='\n', delimiter='\t')
        
        for line in lines:
            new_line = line + list(self.model.wv[line[0]]-self.model.wv[line[1]])
            writer.writerow(new_line)
            
        open_file.close()
        write_file.close()   
    
    def vec_dis(self, open_tsv='metonymy_vector.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        label = [line[0] for line in lines]
        
        vectors = np.array([line[2:] for line in lines])
        dis = np.linalg.norm(vectors, axis=1)
        corr = np.corrcoef(vectors)
        print(corr)
        
        plt.barh(np.arange(len(label)), dis, tick_label=label)
        plt.xlabel('Euclidean distance between metonymy and country')
        plt.savefig('test.png', format='png', dpi=150)
        plt.show()
    
    def mahal(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        metonymy_vectors = [self.model.wv[line[0]] for line in lines]
        country_vectors = [self.model.wv[line[1]] for line in lines]
        
        metonymy_mahal = mahalanobis(np.array(metonymy_vectors,dtype=float))
        country_mahal = mahalanobis(np.array(country_vectors,dtype=float))
        
        return metonymy_mahal, country_mahal
        
    def vec_sim(self, open_tsv='metonymy_vector.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        sims = np.zeros((len(lines), len(lines)))
        for i in range(len(lines)):
            for j in range(len(lines)):
                sims[i][j] = round(cos_sim(np.array(lines[i][2:],dtype=float), np.array(lines[j][2:],dtype=float)),3)
        label = [line[0] for line in lines]
        df = pd.DataFrame(sims, columns=label, index=label)
        return df

# instantiaion
met = Metonymy()