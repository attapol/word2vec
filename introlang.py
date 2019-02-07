import numpy as np
from gensim.models import word2vec 
nor = np.linalg.norm

# load vector data
model = word2vec.Word2Vec.load('article1.model')

# function for searching similar words
def sim1(word, n=5):
    results = model.wv.most_similar(positive=[word], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))
 
# function for calculating the similality of 2 words       
def sim2(word1, word2):
    vec1 = model.wv[word1]
    vec2 = model.wv[word2]
    return round(np.dot(vec1, vec2) / (nor(vec1) * nor(vec2)), 4)

# function for arithmetic
def plus(pos1, pos2, n=5):
    """
    calculate: pos1 + pos2
    
    calc('โตเกียว', 'ญี่ปุ่น', 'จีน')
    >>> ปักกิ่ง
    """
    results = model.wv.most_similar(positive=[pos1, pos2], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))
        
def minus(pos, neg, n=5):
    """
    calculate: pos - neg
    
    calc('โตเกียว', 'ญี่ปุ่น', 'จีน')
    >>> ปักกิ่ง
    """
    results = model.wv.most_similar(positive=[pos], negative=[neg], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))


def calc(pos1, neg1, pos2, n=5):
    """
    calculate: pos1 - neg1 + pos2
    
    calc('โตเกียว', 'ญี่ปุ่น', 'จีน')
    >>> ปักกิ่ง
    """
    results = model.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))
    
    