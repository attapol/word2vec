"""
@author: Nozomi
"""
import numpy as np
from gensim.models import KeyedVectors
nor = np.linalg.norm
import warnings
warnings.filterwarnings('ignore')

# load vector data
model = KeyedVectors.load_word2vec_format('./model.bin', unicode_errors='ignore', binary=True)

# function for searching similar words
def sim1(word, n=5):
    if word not in model:
        print ('{} not in the vocabulary'.format(word))
        return
    results = model.wv.most_similar(positive=[word], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))
 
# function for calculating the similarity of 2 words
def sim2(word1, word2):
    if word1 not in model:
        print ('{} not in the vocabulary'.format(word1))
        return
    if word2 not in model:
        print ('{} not in the vocabulary'.format(word2))
        return
    vec1 = model.wv[word1]
    vec2 = model.wv[word2]
    return round(float(np.dot(vec1, vec2)) / (nor(vec1) * nor(vec2)), 4)

# function for arithmetic
def plus(pos1, pos2, n=5):
    """
    calculate: pos1 + pos2
    
    plus('หนุ่ม', 'ภรรยา')
    > 'สามี'
    """
    results = model.wv.most_similar(positive=[pos1, pos2], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))
        
def minus(pos, neg, n=5):
    """
    calculate: pos - neg
    """
    results = model.wv.most_similar(positive=[pos], negative=[neg], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))

def calc(pos1, neg1, pos2, n=5):
    """
    calculate: pos1 - neg1 + pos2
    
    calc('โตเกียว', 'ญี่ปุ่น', 'จีน')
    > 'ปักกิ่ง'
    """
    results = model.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))
