import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import gensim
import copy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def gensim_doctovec_kmeans(input_df, cpu_cores_to_use, epochs_to_have):
    # This is a work-in-progress implementation of the model that Yuttawee put together up to this point.
    #TODO: (1) It only takes the document column ("review") for now. Investigate a way to add numerical columns as features
    #TODO: (2) Make more hyperparameters accesible by the function itself

    print("Generating Review Tokens -- ")

    input_df['review_tokens'] = input_df['review'].str.lower()
    input_df['review_tokens'] = input_df['review_tokens'].apply(word_tokenize)

    print("Tokens DONE. Using Gensim Doc2Vec and KMeans to do clustering -- ")

    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    all_content_train = []
    j=0
    for em in input_df['review_tokens'].values:
        all_content_train.append(LabeledSentence1(em,[j]))
        j+=1
    print('Number of texts processed: ', j)

    d2v_model = Doc2Vec(all_content_train, vector_size=3, # 100~300
                        window=4, # this is okay but changing it may not affect it
                        min_count=1, # Could bump this up to even 100 to see what happens
                        # Pick max_vocab_size over min_count, as they do the same things
                        # Visualize the loss, if the curve of the loss has yet to reach a plateau shape, then increase the number of epochs
                        workers=cpu_cores_to_use)
    d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=epochs_to_have)

    # Observe the output, plot the loss function, see if the loss jumps around, the the alpha (learning rate) is too high, also tweak min_alpha 

    x_train = d2v_model.dv[0].reshape(1,-1)
    for i in range(1,j):
        x_train = np.append(x_train, d2v_model.dv[i].reshape(1,-1),axis=0)

    kmeans = KMeans(n_clusters=5, random_state=12345).fit(x_train)
    kmeans_labels = kmeans.labels_

    svd = TruncatedSVD(n_components=2, n_iter=10, random_state=12345)
    transformed_unigrams = svd.fit_transform(x_train)

    output_df = copy.deepcopy(input_df)

    output_df["cluster_label"] = kmeans_labels

    print("Clustering DONE. Evaluating using Silhouette Score -- ")

    sil_score = silhouette_score(x_train, kmeans_labels)

    #Visualize the clusters
    x = transformed_unigrams[:,0]
    y = transformed_unigrams[:,1]
    #colors = cm.rainbow(np.linspace(0, 1, y.shape[0]))
    fig, ax = plt.subplots(figsize=(18,10))
    scatter = ax.scatter(x, y, c=kmeans_labels)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax.add_artist(legend1)
    plt.show()

    print("Production is done. Use [0] to see clustered dataframe or [1] for Silhouette Score.")

    return [output_df, sil_score]




