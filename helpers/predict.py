import numpy as np
import torch
import wget
import pandas as pd
import os
import pickle

from transformers import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import *
from pymagnitude import *
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity as cosine
from pymagnitude import *
from operator import itemgetter
from gensim.models import Word2Vec

from helpers.spectral_clustering import *
from config import *


# Load pretrained fasttext model
vectors = Magnitude(os.path.join(DATA_PATH,"wiki-news-300d-1M-subword.magnitude"))

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
    query = query.lower()
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
    embedding = vectors.query(query.replace(',', ' ').split()).mean(axis=0)
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
    query = query.lower()
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
    
def load_models():
    
    with open(os.path.join(GENERATED_DATA_PATH,SPECTRAL_CLUSTERING_FILENAME),'rb') as f:
        spectral_clustering_embed = pickle.load(f)
        ordered_nodes = pickle.load(f)
        name2idx_adjacency = pickle.load(f)
            
    node2vec = Word2Vec.load(os.path.join(GENERATED_DATA_PATH,NODE2VEC_FILENAME))
    
    with open(os.path.join(GENERATED_DATA_PATH,FAST_MEAN_FILENAME),'rb') as f:
        fasttext_dict = pickle.load(f)

    try:
        fasttext_vectors = Magnitude(os.path.join(DATA_PATH,"wiki-news-300d-1M-subword.magnitude"))
    except:
        print("Downloading Magnitude FIle")
        wget.download("http://magnitude.plasticity.ai/fasttext/light/wiki-news-300d-1M-subword.magnitude")
        fasttext_vectors = Magnitude(os.path.join(DATA_PATH,"wiki-news-300d-1M-subword.magnitude"))
        
    df_node = pd.read_csv(os.path.join(GENERATED_DATA_PATH,DF_NODE_FILENAME))

    query_bot = QueryBot(node2vec,spectral_clustering_embed,name2idx_adjacency,ordered_nodes,fasttext_vectors,fasttext_dict,df_node)
        
    return query_bot

def plot_projection(node_vectors, nodes, path, dim=2, projection_method=PCA, perplexity=3):
    """Plot the projection of embeddings after applying a dimensionality reduction method."""
    if projection_method == TSNE:
        projection = projection_method(n_components=dim, perplexity=perplexity)
    else:
        projection = projection_method(n_components=dim)
    projections = projection.fit_transform(node_vectors)
    
    plt.figure(figsize=(10,7))
    plt.scatter(projections[:,0], projections[:,1])
    for i, node in enumerate(nodes):
        plt.annotate(node, xy=(projections[i,0], projections[i,1]))
    plt.title("{} over {} dimensions".format(type(projection).__name__, dim))
    plt.savefig(path)

def get_vectors_spectral(nodes, proj, name2idx_adjacency):
    vectors = []
    for node in nodes:
        idx = name2idx_adjacency[node]
        vector = proj[idx]
        vectors.append(vector.tolist()[0])
    return vectors

def fastt_embedding(text):
    "Encode text using Fasttext model."
    return vectors.query(text.split()).mean(axis=0)

def get_weighted_walk_of_embeddings(walk, embeddings_dict, source_weight=0.75):
    "Given a source embedding and a walk of embeddings, returns the weighted average of embeddings."
    walk_weight = (1-source_weight)/(len(walk)-1)
    embeddings = [embeddings_dict[node] for node in walk]
    return source_weight*embeddings[0] + walk_weight*sum(embeddings[1:])

def walk_averaged_embeddings_dict(embeddings_dict, walks, source_weight=0.75):
    walk_embeddings_dict = {k:[] for k in embeddings_dict.keys()}

    for walk in walks:
        walk_embeddings_dict[walk[0]].append(get_weighted_walk_of_embeddings(walk, embeddings_dict, source_weight))

    walk_averaged_embeddings_dict = {k:sum(v)/len(v) for k, v in walk_embeddings_dict.items()}
    return walk_averaged_embeddings_dict

def get_vectors_fasttext(nodes, walk_averaged_embeddings_dict_fastt):
    vectors = []
    for node in nodes:
        vector = walk_averaged_embeddings_dict_fastt[node]
        vectors.append(vector)
    return vectors
 