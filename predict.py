import numpy as np
from gensim.models import Word2Vec
import torch
from transformers import *
from config import *
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cosine
import os
import pickle
from pymagnitude import *
from operator import itemgetter


class QueryBot:
    def __init__(self, node2vec_model, proj_spectral, fasttext_vectors, fasttext_dict, df_node):
        self.node2vec = node2vec_model
        self.proj_spectral =  proj_spectral
        self.fasttext_vectors = fasttext_vectors
        self.fasttext_dict = fasttext_dict
        self.df_node = df_node

    def make_prediction(self, query, method, topk=TOPK):
        if method == METHODS[0]:
            return query_answers_fasttext(query, self.fasttext_vectors, self.fasttext_dict, self.df_node, topk)
        elif method == METHODS[1]:
            return query_answers_node2vec(query, self.node2vec, self.df_node, topk)
        elif method == METHODS[2]:
            return query_answers_spectral(query, self.proj_spectral, self.df_node, topk)
        else:
            print("Wrong Method")

def compute_similarities_matrix(vector, proj, topk):
    similarities = cosine(vector,proj).reshape(-1)
    top_k_indexes = list(np.argsort(similarities)[-topk:])
    top_k_similarities = similarities[top_k_indexes]
    return dict(zip(top_k_indexes, top_k_similarities))

def query_answers_spectral(query, matrix, df_node, topn=10, return_idx=True):
    filtered = df_node[df_node.name==query].index.values
    if len(filtered)==0:
        return None
    
    idx = filtered[0]
    vector = matrix[idx].reshape(1,-1)
    results = compute_similarities_matrix(vector, matrix, topn)
    
    similarities = {}
    for idx,v in results.items():
        if not return_idx:
            name = df_node[df_node.index==idx].name.values[0]
        else:
            name = idx
            
        similarities[name] = v

    similarities = dict(sorted(similarities.items(), key = itemgetter(1), reverse = True))
    return similarities
    
def compute_similarities(vector, corpus, top_n=10):
    """Given an embedding and a corpus, returns the closest k embeddings."""
    similarities = {k:(cosine(vector.reshape(1,-1), v.reshape(1,-1)))[0,0] for k, v in corpus.items()}
    similarities = dict(sorted(similarities.items(), key = itemgetter(1), reverse = True)[:top_n])
    return similarities

def query_answers_fasttext(query,vectors,walk_averaged_embeddings_dict_fastt,df_node,topn=10, return_idx=True):
    embedding = vectors.query(query.split()).mean(axis=0)
    similarities = compute_similarities(embedding, walk_averaged_embeddings_dict_fastt,topn)
    if return_idx:
        output = {}
        for k,v in similarities.items():
            idx = df_node[df_node.name==k].index.values[0]
            output[idx] = v
        return output
    else:
        return similarities

def query_answers_node2vec(query, model, df_node, topk, return_idx=True):
    splitted_query = query.split(',')
    filtered_query = filter(lambda x: x in model.wv.vocab, splitted_query)
    
    vectors = []
    for w in filtered_query:
        vectors.append(model[w])
            
    if len(vectors)==0:
        return None
      
    vectors = np.array(vectors)
    final_embedding = np.mean(vectors,axis=0)
        
    tuples = model.similar_by_vector(final_embedding,topn=topk)
    
    output_dict = {}
    for word,score in tuples:
        if not return_idx:
            output_dict[word] = score
        else:
            idx = df_node[df_node.name==word].index.values[0]
            output_dict[idx] = score
        
    return output_dict
    
def load_models(folder, spectral_clustering_filename=SPECTRAL_CLUSTERING_FILENAME,fast_mean_filename=FAST_MEAN_FILENAME,node2vec_filename=NODE2VEC_FILENAME,df_node_filename=DF_NODE_FILENAME):
    
    spectral_clustering_embed = np.load(os.path.join(folder,spectral_clustering_filename))
    
    node2vec = Word2Vec.load(os.path.join(folder,node2vec_filename))
    
    with open(os.path.join(folder,fast_mean_filename),'rb') as f:
        fasttext_dict = pickle.load(f)

    fasttext_vectors = Magnitude(os.path.join(folder,"wiki-news-300d-1M-subword.magnitude"))
    
    df_node = pd.read_csv(os.path.join(folder,df_node_filename))

    query_bot = QueryBot(node2vec,spectral_clustering_embed,fasttext_vectors,fasttext_dict,df_node)
        
    return query_bot


 