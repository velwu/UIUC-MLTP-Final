# Vel: Doing some word cloud because I got curious
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import operator
import string
import itertools
import operator
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

def get_top_words(dataset_with_text, m, n):
    top_words_origin = dataset_with_text["review"].str.replace(","," ").str.cat().split()
    top_words_base = {}
    for each_word in top_words_origin:
        if each_word in top_words_base: 
            top_words_base[each_word] += 1 
        else:
            top_words_base[each_word] = 1
    top_words_base = sorted(top_words_base.items(), key=operator.itemgetter(1))
    top_words_base = dict(top_words_base[::-1])
    
    top_words_base_top_n = {k: top_words_base[k] for k in list(top_words_base)[m:n]} 


    top_words_base_no_punc_words = {}
    for word, value in top_words_base_top_n.items():
        if word[-1] in string.punctuation: 
            word = word[:-1] 
        if len(word) > 0: 
            if word in top_words_base_no_punc_words: 
                top_words_base_no_punc_words[word] += value 
            else: 
                top_words_base_no_punc_words[word] = value
    top_words_base_no_punc_sorted = sorted(top_words_base_no_punc_words.items(), 
                                           key=operator.itemgetter(1)) 
    top_words_dict_final = dict(top_words_base_no_punc_sorted[::-1])
    return top_words_dict_final

def get_top_words_wo_stopwords(dataset_with_text, m, n):
    top_words_origin = dataset_with_text["review"].str.replace(","," ").str.cat().split()
    top_words_base = {}
    for each_word in top_words_origin:
        if each_word in top_words_base: 
            top_words_base[each_word] += 1 
        else:
            top_words_base[each_word] = 1
    top_words_base = sorted(top_words_base.items(), key=operator.itemgetter(1))
    top_words_base = dict(top_words_base[::-1])
    
    stop_words = set(stopwords.words('english'))
    top_words_base_less_words = {}
    for w, num in top_words_base.items():
        if w.lower() not in stop_words:
            top_words_base_less_words[w] = num
    
    top_words_base_top_n = {k: top_words_base_less_words[k] for k in list(top_words_base_less_words)[m:n]} 

    top_words_base_no_punc_words = {}
    for word, value in top_words_base_top_n.items():
        if word[-1] in string.punctuation: 
            word = word[:-1]
        if word[0] in string.punctuation:
            word = word[1:]
        if len(word) > 0: 
            if word in top_words_base_no_punc_words: 
                top_words_base_no_punc_words[word] += value 
            else: 
                top_words_base_no_punc_words[word] = value
    top_words_base_no_punc_sorted = sorted(top_words_base_no_punc_words.items(), 
                                           key=operator.itemgetter(1)) 
    top_words_dict_final = dict(top_words_base_no_punc_sorted[::-1])
    return top_words_dict_final

def plot_top_words(df_to_plot, param, fig_x_size, fig_y_size, fig_title, allow_stopwords):
# param =  Most used words in all data
    if allow_stopwords == True:
        top_words_dict = get_top_words(df_to_plot, 0, param)
    else: 
        top_words_dict = get_top_words_wo_stopwords(df_to_plot, 0, param)
    
    [print("Word:", key, "; Occurrence:", value) for key, value in top_words_dict.items()]
    keys = list(top_words_dict.keys())
    # get values in the same order as keys, and parse percentage values
    vals = [float(top_words_dict[k] / len(df_to_plot) * 100) for k in keys]
    fig = plt.figure(figsize=(fig_x_size, fig_y_size))
    fig.suptitle('Top Words in Document: '+ fig_title, fontsize=20)
    plt.xlabel('Top Words', fontsize=18)
    plt.ylabel('Probability of Occurrence', fontsize=16)
    sns.barplot(x=keys, y=vals)
    
    plt.savefig(fig_title +'.png')