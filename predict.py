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
    def __init__(self, node2vec_model, proj_spectral, name2idx_adjacency, ordered_nodes, fasttext_vectors, fasttext_dict, df_node):
        self.node2vec = node2vec_model
        
        self.proj_spectral =  proj_spectral
        self.name2idx_adjacency = name2idx_adjacency
        self.ordered_nodes = ordered_nodes
        
        self.fasttext_vectors = fasttext_vectors
        self.fasttext_dict = fasttext_dict
        
        self.df_node = df_node

    def make_prediction(self, query, method, topk=TOPK):
        if method == METHODS[0]:
            return query_answers_fasttext(query, self.fasttext_vectors, self.fasttext_dict, self.df_node, topk)
        elif method == METHODS[1]:
            return query_answers_node2vec(query, self.node2vec, self.df_node, topk)
        elif method == METHODS[2]:
            return query_answers_spectral(query, self.proj_spectral, self.ordered_nodes, self.df_node, self.name2idx_adjacency, topk)
        else:
            print("Wrong Method")

def compute_similarities_matrix(vector, proj, topk):
    similarities = cosine(vector,proj).reshape(-1)
    top_k_indexes = list(np.argsort(similarities)[-topk:])
    top_k_similarities = similarities[top_k_indexes]
    return dict(zip(top_k_indexes, top_k_similarities))

def query_answers_spectral(query, matrix, ordered_nodes, df_node, name2idx_adjacency, topn=10, return_idx=True):
    splitted_query = query.split(',')
    filtered_query = []
    for x in splitted_query:
        filtered = df_node[df_node.name==x]
        if len(filtered)!=0:
            filtered_query.append(x)
            
    if len(filtered_query)==0:
        return None
    
    vectors = []
    for w in filtered_query:
        idx = name2idx_adjacency[w]
        vectors.append(matrix[idx])
    
    final_embedding = np.mean(vectors,axis=0)

    results = compute_similarities_matrix(final_embedding, matrix, topn)
    
    similarities = {}
    for idx,v in results.items():
        if return_idx:
            name = ordered_nodes[idx]
            idx = df_node[df_node.name==name].index.values[0]
            similarities[idx] = v
        else:
            name = ordered_nodes[idx]
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
    
    with open(os.path.join(folder,spectral_clustering_filename),'rb') as f:
        spectral_clustering_embed = pickle.load(f)
        ordered_nodes = pickle.load(f)
        name2idx_adjacency = pickle.load(f)
            
    node2vec = Word2Vec.load(os.path.join(folder,node2vec_filename))
    
    with open(os.path.join(folder,fast_mean_filename),'rb') as f:
        fasttext_dict = pickle.load(f)

    fasttext_vectors = Magnitude(os.path.join(folder,"wiki-news-300d-1M-subword.magnitude"))
    
    df_node = pd.read_csv(os.path.join(folder,df_node_filename))

    query_bot = QueryBot(node2vec,spectral_clustering_embed,name2idx_adjacency,ordered_nodes,fasttext_vectors,fasttext_dict,df_node)
        
    return query_bot


 