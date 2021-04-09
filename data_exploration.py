import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import string
import re
import os
import csv
import string
import itertools
import operator
import re
import copy

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.cluster import KMeans

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

def draw_word_cloud_from_dataframe(data_passed_in, cluster_num, cloud_width, cloud_height):
    if cluster_num == "All": 
        data_to_plot = data_passed_in
    else:
        data_to_plot = data_passed_in[data_passed_in["cluster_label"] == cluster_num]
    word_cloud_origin = data_to_plot["review"].str.replace(","," ").str.cat().split()
    word_cloud_base = {}
    for each_word in word_cloud_origin:
        if each_word in word_cloud_base: 
            word_cloud_base[each_word] += 1 
        else:
            word_cloud_base[each_word] = 1
    word_cloud_base = sorted(word_cloud_base.items(), key=operator.itemgetter(1))
    word_cloud_base = dict(word_cloud_base[::-1])
    stop_words = set(stopwords.words('english'))
    word_cloud_base_less_words = {}
    for w, num in word_cloud_base.items():
        if w.lower() not in stop_words:
            word_cloud_base_less_words[w] = num
    word_cloud_base_no_punc_words = {}
    for word, value in word_cloud_base_less_words.items():
        if len(word) > 0: 
            if word[-1] in string.punctuation: 
                word = word[:-1]
        if len(word) > 0: 
            if word[0] in string.punctuation:
                word = word[1:]
        if len(word) > 0: 
            if word in word_cloud_base_no_punc_words: 
                word_cloud_base_no_punc_words[word] += value 
            else: 
                word_cloud_base_no_punc_words[word] = value
    word_cloud_base_no_punc_sorted = sorted(word_cloud_base_no_punc_words.items(), key=operator.itemgetter(1)) 
    word_cloud_base_no_punc_sorted = dict(word_cloud_base_no_punc_sorted[::-1])
    word_cloud_main = WordCloud().generate_from_frequencies(word_cloud_base_no_punc_sorted)
    fig, ax = plt.subplots(figsize=(cloud_width,cloud_height))
    ax.imshow(word_cloud_main)
    
    fig_title = "Wordcloud - Cluster " + str(cluster_num)
    plt.savefig(fig_title +'.png')
    
    plt.show()

def draw_word_clouds_for_clusters(df_clustered, num_o_clusters):
    for each_iteration in list(range(num_o_clusters)):
        print("Cluster " + str(each_iteration))
        draw_word_cloud_from_dataframe(df_clustered, int(each_iteration), 16, 9)

def KMeans_and_SVD(clusters_count, initial_n, df_to_be_clustered):
    
    df_clustered = copy.deepcopy(df_to_be_clustered)
    
    vec_of_vel = TfidfVectorizer(stop_words='english')
    vec_of_vel.fit(df_clustered.review)
    features_of_vec = vec_of_vel.transform(df_clustered.review)
    clusters_of_vel = KMeans(init = "k-means++", n_clusters = clusters_count, n_init = initial_n, random_state = 557)
    clusters_of_vel.fit(features_of_vec)
    df_clustered["cluster_label"] = clusters_of_vel.labels_

    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    # https://stackoverflow.com/questions/47006268/matplotlib-scatter-plot-with-color-label-and-legend-specified-by-c-option
    svd_tryout = TruncatedSVD(n_components=2)
    # svd_tryout.fit(features_of_vec)
    xxx_svd = svd_tryout.fit_transform(features_of_vec)
    y_kmeans = clusters_of_vel.predict(features_of_vec)
    svd_cluster_sizes = pd.value_counts(y_kmeans)
    svd_cluster_sizes.sort_values
    svd_cluster_sizes.plot.bar
    # centers = clusters_of_vel.cluster_centers_
    # plt.scatter(xxx_svd[:, 0], xxx_svd[:, 1], c=y_kmeans, s=50, cmap='viridis')
    fig, ax = plt.subplots(figsize=(16,9))
    for g in np.unique(y_kmeans):
        ix = np.where(y_kmeans == g)
        ax.scatter(xxx_svd[ix, 0], xxx_svd[ix, 1], label = g, s = 7)
    ax.legend()
    ax.grid(True)
    plt.title(str(clusters_count) + " Clusters from Truncated SVD", fontsize=40)
    plt.show()
    
    return [df_clustered, features_of_vec, clusters_of_vel, xxx_svd, y_kmeans]