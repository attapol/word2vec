#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:37:09 2019

@author: Nozomi
"""

from cutkum.tokenizer import Cutkum
import nltk
import tltk
from pythainlp.tokenize import word_tokenize

ck = Cutkum()

def token(text):
    
    print(ck.tokenize(text))
    print('\n')
    print(tltk.nlp.word_segment(text))
    print('\n')
    print(word_tokenize(text))