import numpy as np
from gensim.models import Word2Vec
import torch
from transformers import *
from config import *
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cosine

class Bert_embedder:
    def __init__(self, tokenizer, model, node_embeddings):
        self.tokenizer = tokenizer
        self.model = model
        self.node_embeddings = node_embeddings
        
    def compute_similarities(self, vector, topk):
        similarities = cosine(vector,self.node_embeddings)
        top_k_indexes = np.argsort(similarities)[-topk:]
        top_k_similarities = similarities[top_k_indexes]
        return dict(zip(top_k_indexes, top_k_similarities))
        
    def find_most_similar_articles(query,topk):
        # Encode text
        input_ids = torch.tensor([tokenizer.encode(query, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
        vector = last_hidden_states.reshape(1,-1)
        
        return compute_similarities(vector,topk)
    
def process_query_node2vec(query, model, topk, df_node):
    splitted_query = query.split(',')
    filtered_query = filter(lambda x: x in model.vocab, splitted_word)
    
    vectors = []
    for w in filtered_query:
        vectors.append(model[w])
        
    final_embedding = np.mean(vectors)
    
    tuples = model.similar_by_vector(final_embedding,topk)
    
    output_dict = {}
    for word,score in tuples:
        idx = df_node[df_node['name']==word].index.values.astype(int)[0]
        output_dict[idx] = score
        
    return output_dict
    
def load_models(spectral_clustering_path=SPECTRAL_CLUSTERING_PATH,bert_mean_path=BERT_MEAN_PATH,node2vec_path=NODE2VEC_PATH,df_node_path=DF_NODE_PATH):
    spectral_clustering_embed = np.load(spectral_clustering_path)
    bert_mean_embed = np.load(bert_mean_path)
    node2vec_embed = Word2Vec.load(node2vec_path)
    df_node = pd.read_csv(df_node_path)
    
    pretrained_weights = 'bert-base-uncased'
    tokenizer_bert = BertTokenizer.from_pretrained(pretrained_weights)
    model_bert = BertModel.from_pretrained(pretrained_weights)
    
    bert_embedder_spectral = Bert_embedder(tokenizer_bert,model_bert,spectral_clustering_embed)
    bert_embedder_mean = Bert_embedder(tokenizer_bert,model_bert,bert_mean_embed)
    
    return bert_embedder_spectral, bert_embedder_mean, node2vec_embed, df_node

def make_prediction(query, method, bert_embedder_spectral, bert_embedder_mean, node2vec_embed, df_node, topk=TOPK):
    if method == methods[0]:
        return bert_embedder_spectral.find_most_similar_articles(query,topk)
    elif method == methods[1]:
        return bert_embedder_mean.find_most_similar_articles(query,topk)
    elif method == methods[2]:
        return process_query_node2vec(query,node2vec_embed,topk,df_node)       
 