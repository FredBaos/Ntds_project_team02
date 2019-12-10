import numpy as np
from gensim.models import Word2Vec
import torch
from transformers import *
from config import *
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cosine
import os

class Bert_embedder:
    def __init__(self, tokenizer, model, node_embeddings):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.node_embeddings = node_embeddings
        
    def compute_similarities(self, vector, topk):
        similarities = cosine(vector,self.node_embeddings).reshape(-1)
        top_k_indexes = list(np.argsort(similarities)[-topk:])
        top_k_similarities = similarities[top_k_indexes]
        return dict(zip(top_k_indexes, top_k_similarities))
        
    def find_most_similar_articles(self,query,topk):
        # Encode text
        input_ids = torch.tensor([self.tokenizer.encode(query, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
            last_hidden_states = last_hidden_states.squeeze().mean(dim=0).numpy()
            
        vector = last_hidden_states.reshape(1,-1)
        
        return self.compute_similarities(vector,topk)
    
def process_query_node2vec(query, model, topk, df_node):
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
        idx = df_node[df_node['name']==word].index.values.astype(int)[0]
        output_dict[idx] = score
        
    return output_dict
    
def load_models(folder, spectral_clustering_filename=SPECTRAL_CLUSTERING_FILENAME,bert_mean_filename=BERT_MEAN_FILENAME,node2vec_filename=NODE2VEC_FILENAME,df_node_filename=DF_NODE_FILENAME):
    
    spectral_clustering_embed = np.load(os.path.join(folder,spectral_clustering_filename))
    bert_mean_embed = np.load(os.path.join(folder,bert_mean_filename))
    node2vec_embed = Word2Vec.load(os.path.join(folder,node2vec_filename))
    df_node = pd.read_csv(os.path.join(folder,df_node_filename))
    
    pretrained_weights = 'bert-base-uncased'
    tokenizer_bert = BertTokenizer.from_pretrained(pretrained_weights)
    model_bert = BertModel.from_pretrained(pretrained_weights)
    
    bert_embedder_spectral = Bert_embedder(tokenizer_bert,model_bert,spectral_clustering_embed)
    bert_embedder_mean = Bert_embedder(tokenizer_bert,model_bert,bert_mean_embed)
    
    return bert_embedder_spectral, bert_embedder_mean, node2vec_embed, df_node

def make_prediction(query, method, bert_embedder_spectral, bert_embedder_mean, node2vec_embed, df_node, topk=TOPK):
    if method == METHODS[0]:
        return bert_embedder_spectral.find_most_similar_articles(query,topk)
    elif method == METHODS[1]:
        return bert_embedder_mean.find_most_similar_articles(query,topk)
    elif method == METHODS[2]:
        return process_query_node2vec(query,node2vec_embed,topk,df_node)
    else:
        print("Wrong Method")
 