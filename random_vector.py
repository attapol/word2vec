# -*- coding: utf-8 -*-
"""
@author: Nozomi
"""
import numpy as np
import matplotlib.pyplot as plt

# function for cosine similarity
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# plot histogram 
def sim_distribution(dim, k=1000):
    # make random k vectors that have n dimensions
    # randn = Gaussian
    vectors = np.random.randn(k, dim)

    sims = []
    for i in range(k):
        for j in range(i+1, k):
            sims.append(cos_sim(vectors[i], vectors[j]))
    
    count = 0
    for i in sims:
        if i >= 0.5:
            count += 1
    print(count,len(sims))
    
    plt.hist(sims, bins=200, range=(-1,1))
    plt.xlabel('cosine similarity')
    plt.ylabel('Numbers')
    plt.title('cosine similarity distribution on {} dim, {} samples'.format(dim,k))
    plt.show()
    
def dis_distribution(dim, k=1000):
    # make random k vectors that have n dimensions
    # randn = Gaussian
    vectors = np.random.randn(k, dim)

    dis = []
    for i in range(k):
        for j in range(i+1, k):
            dis.append(np.linalg.norm(vectors[i]- vectors[j]))
    
    plt.hist(dis, bins=200, range=(0,50))
    plt.xlabel('Euclidean distance')
    plt.ylabel('Numbers')
    plt.title('Euclidean distance distribution on {} dim, {} samples'.format(dim,k))
    plt.show()