import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from nltk.tokenize import word_tokenize

import ipywidgets
import data_pre_processing as file_dpp

def model_prep(use_num_clusters):

    df_list = file_dpp.main_proc(use_num_clusters)
    merged_game_sale_and_reviews = df_list[-1]

    merged_game_sale_and_reviews['review_tokens'] = merged_game_sale_and_reviews['review'].str.lower()
    merged_game_sale_and_reviews['review_tokens'] = merged_game_sale_and_reviews['review_tokens'].apply(word_tokenize)

    #Give tag numbers to each review
    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    all_content_train = []
    j=0
    for em in merged_game_sale_and_reviews['review_tokens'].values:
        all_content_train.append(LabeledSentence1(em,[j]))
        j+=1
    print('Number of texts processed: ', j)

    return [all_content_train, merged_game_sale_and_reviews]

#Create a function to construct document embeddings
def get_doc2vec(tagged_doc, param_list):
    vector_size = param_list[0]
    window = param_list[1]
    min_count = param_list[2]
    negative = param_list[3]
    epoch = param_list[4]
    
    d2v_model = Doc2Vec(tagged_doc, vector_size=vector_size, window=window, min_count=min_count, 
                        workers=4, negative=negative)
    d2v_model.train(tagged_doc, total_examples=d2v_model.corpus_count, epochs=epoch)
    
    return d2v_model

# Function to add a column to Doc2Vec embeddings
def append_col_to_d2v(d2v_model,added_col):
    '''
    
    :param d2v_model: Doc2Vec embeddings 
    :param added_col: A column that the user want to add to the embedding, should be a DataFrame, series, or numpy array, and has 1D
    :return: A new Doc2Vec embeddings that is already added the additional column 
    '''
    added_col_arr = np.array(added_col).reshape(-1,1)
    d2v_model.dv.vectors = np.append(d2v_model.dv.vectors,added_col_arr,axis=1)
    
    return d2v_model

def run_recommender(d2v_model, merged_game_sale_and_reviews):
    input_game_name = input('Enter game name: ')
    input_game_name = input_game_name.lower()
    input_game_name = input_game_name.strip()
    print(input_game_name)
    #Get input game score from the user, clean it, and change the format to float
    input_game_score = input('Enter game score (0.0-10.0): ')
    input_game_score = input_game_score.strip()
    input_game_score = float(input_game_score)
    print(input_game_score)

    #Filter the df to get only the rows that contain the input game name 
    filtered_df = merged_game_sale_and_reviews[merged_game_sale_and_reviews['game'].str.contains(input_game_name)]
    unique_user_score = filtered_df['User_Score'].unique()
    print(unique_user_score)
    dif_array = np.abs(unique_user_score - input_game_score)
    closest_user_score = unique_user_score[dif_array.argmin()]

    #Filter the df again to get the rows that contain the closest game score
    filtered_df = filtered_df[filtered_df['User_Score']==closest_user_score]

    #Get the following embeddings from the filtered_df indexes 
    filtered_d2v = d2v_model.dv[filtered_df.index]

    #Average the filtered embeddings to get an average vector in the vector space
    avg_filtered_d2v = np.average(filtered_d2v, axis=0)

    #Get the most 10 similar embeddings (compared to the avg filtered embedding) 
    similar_reviews = d2v_model.docvecs.most_similar([avg_filtered_d2v])

    #Print out the inputs and the recommended games
    game_idx = [i[0] for i in similar_reviews]
    print(game_idx)
    game_similarity = [i[1] for i in similar_reviews]
    print(game_similarity)
    recommended_games = merged_game_sale_and_reviews.loc[game_idx,:]['game'].unique()

    print(' ')
    print('Input game: ' + input_game_name)
    print('Input user score: ' + str(input_game_score))
    print('Similarity - Recommended games')
    for i, j in zip(recommended_games, game_similarity):
        print(str(j) + ' - ' + i)

if __name__ == "__main__":
    print("Warning: This file is intended to be called by another program, not run by itself.")